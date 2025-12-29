import os
import json
import random
import re
from PIL import Image

import numpy as np
import torch
from numpy.polynomial.chebyshev import chebval
from tqdm import tqdm

from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from pycocotools import mask as maskUtils
from open_r1.utils.pycocotools.coco import COCO
from open_r1.utils.pycocotools.cocoeval import COCOeval


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def cheby_to_polygon(centre, coeffs, n_points: int = 72):
    """Decode radial Chebyshev coefficients into a polygon list."""
    cx, cy = centre
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False, dtype=np.float32)
    radii = chebval(np.cos(thetas), coeffs)
    radii = np.maximum(radii, 0.0)
    xs = cx + radii * np.cos(thetas)
    ys = cy + radii * np.sin(thetas)
    poly = []
    for x, y in zip(xs, ys):
        poly.extend([float(x), float(y)])
    return poly


def extract_mask_answer(content):
    """Return polygon coordinates parsed from `<answer>` JSON."""
    match = ANSWER_RE.search(content)
    if not match:
        return None
    json_match = re.search(r"\{.*\}", match.group(1), re.DOTALL)
    if not json_match:
        return None
    try:
        obj = json.loads(json_match.group(0))
    except Exception:
        return None

    if "centre" in obj and "coeffs" in obj:
        return cheby_to_polygon(obj["centre"], obj["coeffs"])
    return obj.get("polygon") or obj.get("polygons")


def load_model(model_path, device_map):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def eval_seg_r1(model_path, test_datasets, data_root, image_root, question_template, output_dir, batch_size=32, sample_num=500, seed=42, device_map="cuda:0"):
    random.seed(seed)
    model, processor = load_model(model_path, device_map)

    for ds in test_datasets:
        print(f"Processing {ds}...")
        ds_path = os.path.join(data_root, f"{ds}.jsonl")
        with open(ds_path, "r") as f:
            data = [json.loads(line) for line in f]
        random.shuffle(data)
        data = data[:sample_num]
        messages = []
        for x in data:
            image_path = os.path.join(image_root, x["image"])
            if "normal_caption" in x:
                q_text = question_template.format(Question=x["normal_caption"])
            else:
                q_text = x["conversations"][0]["value"].replace("<image>", "")

            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": q_text}
                    ]
                }
            ])

        all_outputs = []
        for i in tqdm(range(0, len(messages), batch_size)):
            batch_messages = messages[i:i + batch_size]
            text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(device_map)
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            batch_output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            all_outputs.extend(batch_output_text)

        gt_anns = []
        dt_anns = []
        images = []
        for idx, (ex, out) in enumerate(zip(data, all_outputs)):
            sol_match = ANSWER_RE.search(ex["conversations"][1]["value"])
            if not sol_match:
                continue
            sol_json = re.search(r"\{.*\}", sol_match.group(1), re.DOTALL)
            if not sol_json:
                continue
            sol = json.loads(sol_json.group(0))

            size = sol.get("size")
            if size is not None:
                width, height = size[1], size[0]
            else:
                image = Image.open(os.path.join(image_root, ex["image"]))
                width, height = image.size

            if "centre" in sol and "coeffs" in sol:
                gt_poly = cheby_to_polygon(sol["centre"], sol["coeffs"])
            else:
                gt_poly = sol.get("polygon") or sol.get("polygons")

            pred_poly = extract_mask_answer(out) or []

            gt_polys = gt_poly if isinstance(gt_poly, list) and isinstance(gt_poly[0], list) else [gt_poly]
            gt_rle = maskUtils.merge(maskUtils.frPyObjects(gt_polys, height, width))

            if pred_poly:
                pred_polys = pred_poly if isinstance(pred_poly, list) and isinstance(pred_poly[0], list) else [pred_poly]
                pred_rle = maskUtils.merge(maskUtils.frPyObjects(pred_polys, height, width))
            else:
                pred_rle = {"size": [height, width], "counts": ""}
            if isinstance(gt_rle["counts"], bytes):
                gt_rle["counts"] = gt_rle["counts"].decode("utf-8")
            if isinstance(pred_rle.get("counts"), bytes):
                pred_rle["counts"] = pred_rle["counts"].decode("utf-8")

            images.append({"id": idx, "width": width, "height": height})
            gt_anns.append({"id": idx, "image_id": idx, "category_id": 1, "segmentation": gt_rle, "area": float(maskUtils.area(gt_rle)), "iscrowd": 0})
            dt_anns.append({"id": idx, "image_id": idx, "category_id": 1, "segmentation": pred_rle, "score": 1.0})

        coco_gt = COCO()
        coco_gt.dataset = {"images": images, "annotations": gt_anns, "categories": [{"id": 1, "name": "segm"}]}
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dt_anns)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        result_path = os.path.join(output_dir, f"{os.path.basename(model_path)}", f"{ds}_seg_r1.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        json.dump(coco_eval, open(result_path, "w"), indent=2)
        print(f"Results saved to {result_path}")
        print('-' * 100)


if __name__ == "__main__":
    model_path = '/workspace/VLM-R1/checkpoints/rl/Qwen2.5-VL-3B-Instruct-segment/checkpoint-1300'
    data_root = '/workspace/VLM-R1/data'
    test_datasets = ['val']
    image_root = '/workspace/VLM-R1/data'
    output_dir = 'logs'
    device_map = 'cuda:0'
    question_template = '{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format.'
    eval_seg_r1(model_path, test_datasets, data_root, image_root, question_template, output_dir, device_map=device_map)
