import os
import json
import random
import re
from PIL import Image

import numpy as np
import torch
from tqdm import tqdm

from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_bbox_answer(content):
    """Return a [x1, y1, x2, y2] list parsed from `<answer>` JSON."""
    match = ANSWER_RE.search(content)
    if not match:
        return None
    answer_text = match.group(1).strip()
    try:
        obj = json.loads(answer_text)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, list) and len(obj) == 4 and all(isinstance(x, (int, float)) for x in obj):
        return [float(x) for x in obj]
    bbox_match = re.search(r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]", answer_text)
    if bbox_match:
        return [float(bbox_match.group(i)) for i in range(1, 5)]
    return None


def resize_bbox(bbox, input_height, input_width, image_height, image_width):
    bbox[0] = bbox[0] / input_width * image_width
    bbox[1] = bbox[1] / input_height * image_height
    bbox[2] = bbox[2] / input_width * image_width
    bbox[3] = bbox[3] / input_height * image_height
    return bbox


def load_model(model_path, device_map):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


_SAM2_PREDICTOR = None


def get_sam2_predictor():
    global _SAM2_PREDICTOR
    if _SAM2_PREDICTOR is None:
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = os.getenv("SAM2_DEVICE", "cuda")
        sam2_model = build_sam2_hf("facebook/sam2.1-hiera-large", device=device)
        _SAM2_PREDICTOR = SAM2ImagePredictor(sam2_model)
    return _SAM2_PREDICTOR


def resolve_seg_mask_path(mask_path):
    if mask_path is None:
        raise ValueError("seg_mask_path missing in dataset item for SAM-based evaluation.")
    if os.path.isabs(mask_path):
        return mask_path
    seg_mask_root = os.getenv("SEG_MASK_ROOT")
    if seg_mask_root is None:
        raise ValueError("SEG_MASK_ROOT environment variable must be set when seg_mask_path is relative.")
    return os.path.join(seg_mask_root, mask_path)


def load_gt_mask(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    return mask > 0


def mask_iou(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return inter / union if union else 0.0


def sam_mask_from_bbox(image_path, bbox):
    predictor = get_sam2_predictor()
    np_image = np.array(Image.open(image_path).convert("RGB"))
    predictor.set_image(np_image)
    masks, _, _ = predictor.predict(
        box=np.array(bbox, dtype=np.float32)[None, :],
        multimask_output=False,
    )
    return masks[0] > 0


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
            elif "problem" in x:
                q_text = question_template.format(Question=x["problem"])
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
            for idx, output_text in enumerate(batch_output_text):
                input_height = int(inputs["image_grid_thw"][idx][1] * 14)
                input_width = int(inputs["image_grid_thw"][idx][2] * 14)
                all_outputs.append({
                    "text": output_text,
                    "input_height": input_height,
                    "input_width": input_width,
                })

        results = []
        total_iou = 0.0
        correct_number = 0
        for ex, out in zip(data, all_outputs):
            image_path = os.path.join(image_root, ex["image"])
            image = Image.open(image_path)
            image_width, image_height = image.size

            pred_bbox = extract_bbox_answer(out["text"]) or [0.0, 0.0, 0.0, 0.0]
            resized_bbox = resize_bbox(pred_bbox, out["input_height"], out["input_width"], image_height, image_width)

            seg_mask_path = resolve_seg_mask_path(ex.get("seg_mask_path"))
            gt_mask = load_gt_mask(seg_mask_path)

            mask_score = 0.0
            correct = 0
            try:
                pred_mask = sam_mask_from_bbox(image_path, resized_bbox)
                mask_score = mask_iou(pred_mask, gt_mask)
                if mask_score > 0.5:
                    correct = 1
            except Exception as exc:
                print(f"SAM inference failed for {image_path}: {exc}")

            total_iou += mask_score
            correct_number += correct
            results.append({
                "image": ex["image"],
                "question": ex.get("normal_caption") or ex.get("problem") or ex["conversations"][0]["value"],
                "seg_mask_path": seg_mask_path,
                "model_output": out["text"],
                "input_size": (out["input_height"], out["input_width"]),
                "image_size": (image_height, image_width),
                "extracted_answer": resized_bbox,
                "mask_iou": mask_score,
                "correct": correct,
            })

        mean_iou = total_iou / len(results) if results else 0.0
        accuracy = correct_number / len(results) * 100 if results else 0.0
        print(f"\nMean IoU of {ds}: {mean_iou:.4f}")
        print(f"Accuracy of {ds}: {accuracy:.2f}%")

        result_path = os.path.join(output_dir, f"{os.path.basename(model_path)}", f"{ds}_seg_r1.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump({
                "mean_iou": mean_iou,
                "accuracy": accuracy,
                "results": results,
            }, f, indent=2)
        print(f"Results saved to {result_path}")
        print('-' * 100)


if __name__ == "__main__":
    model_path = '/workspace/VLM-R1/checkpoints/rl/Qwen2.5-VL-3B-Instruct-segment/checkpoint-1300'
    data_root = '/workspace/VLM-R1/data'
    test_datasets = ['val']
    image_root = '/workspace/VLM-R1/data'
    output_dir = 'logs'
    device_map = 'cuda:0'
    question_template = '{Question} First, write your reasoning inside <think>...</think> tags. Next, write the final answer inside <answer>...</answer> tags. The content of <answer> MUST be a valid JSON list of four numbers [x1, y1, x2, y2] in image coordinates. Use exactly that list and no other keys or text.'
    eval_seg_r1(model_path, test_datasets, data_root, image_root, question_template, output_dir, device_map=device_map)
