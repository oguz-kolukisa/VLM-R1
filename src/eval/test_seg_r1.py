import os
import json
import random
import re
import torch
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from pycocotools import mask as maskUtils
from open_r1.utils.pycocotools.coco import COCO
from open_r1.utils.pycocotools.cocoeval import COCOeval


def extract_mask_answer(content):
    pattern = r'```json(.*?)```'
    json_match = re.search(pattern, content, re.DOTALL)
    mask_json = json_match.group(1).strip() if json_match else None
    if mask_json:
        try:
            return json.loads(mask_json)[0]["mask"]
        except Exception:
            return None
    return None


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
        ds_path = os.path.join(data_root, f"{ds}.json")
        data = json.load(open(ds_path, "r"))
        random.shuffle(data)
        data = data[:sample_num]
        messages = []
        for x in data:
            image_path = os.path.join(image_root, x["image"])
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": question_template.format(Question=x["normal_caption"])}
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
            gt_mask = ex["solution"]
            pred_mask = extract_mask_answer(out)
            if pred_mask is None:
                pred_mask = {"size": gt_mask["size"], "counts": ""}
            width, height = gt_mask["size"][1], gt_mask["size"][0]
            images.append({"id": idx, "width": width, "height": height})
            gt_anns.append({"id": idx, "image_id": idx, "category_id": 1, "segmentation": gt_mask, "area": float(maskUtils.area(gt_mask)), "iscrowd": 0})
            dt_anns.append({"id": idx, "image_id": idx, "category_id": 1, "segmentation": pred_mask, "score": 1.0})

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
        json.dump({"AP": coco_eval.stats[0]}, open(result_path, "w"), indent=2)
        print(f"Results saved to {result_path}")
        print('-' * 100)


if __name__ == "__main__":
    model_path = ''
    data_root = ''
    test_datasets = ['val']
    image_root = ''
    output_dir = 'logs'
    device_map = 'cuda:0'
    question_template = '{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format.'
    eval_seg_r1(model_path, test_datasets, data_root, image_root, question_template, output_dir, device_map=device_map)
