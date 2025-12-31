import os
import sys
import json
import random
import re
from typing import Iterable, List
from PIL import Image

import numpy as np
import torch
from tqdm import tqdm

from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Ensure we can import the shared VLM module definitions used during GRPO.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN_R1_SRC = os.path.join(REPO_ROOT, "open-r1-multimodal", "src")
if OPEN_R1_SRC not in sys.path:
    sys.path.append(OPEN_R1_SRC)

from open_r1.vlm_modules.qwen_module import Qwen2VLModule  # noqa: E402


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def strtobool_env(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


def repeat_indices(indices: Iterable[int], repeat: int) -> List[int]:
    return [idx for idx in indices for _ in range(repeat)]


def repeat_batch_encoding(batch_encoding, repeat: int):
    if repeat == 1:
        return batch_encoding
    for key, value in list(batch_encoding.items()):
        if isinstance(value, torch.Tensor):
            batch_encoding[key] = value.repeat_interleave(repeat, dim=0)
        elif isinstance(value, list):
            batch_encoding[key] = [elem for elem in value for _ in range(repeat)]
    return batch_encoding


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


def dataset_name_from_path(path):
    parent = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    return parent if stem == "metadata" else stem


def resolve_dataset_entries(test_datasets, data_root, data_paths_env):
    if data_paths_env:
        data_paths = [p for p in data_paths_env.split(":") if p]
        return [(dataset_name_from_path(p), p) for p in data_paths]

    dataset_entries = []
    for ds in test_datasets:
        ds = ds.strip()
        if not ds:
            continue
        if ds.endswith(".jsonl") and os.path.isfile(ds):
            dataset_entries.append((dataset_name_from_path(ds), ds))
            continue
        candidate_dir = os.path.join(data_root, ds)
        if os.path.isdir(candidate_dir):
            candidate_path = os.path.join(candidate_dir, "metadata.jsonl")
        else:
            candidate_path = os.path.join(data_root, f"{ds}.jsonl")
        if not os.path.isfile(candidate_path):
            raise FileNotFoundError(f"Could not find dataset file for {ds} under {data_root}")
        dataset_entries.append((ds, candidate_path))
    return dataset_entries


def eval_seg_r1(
    model_path,
    test_datasets,
    data_root,
    image_root,
    question_template,
    output_dir,
    batch_size=32,
    sample_num=500,
    seed=42,
    device_map="cuda:0",
    data_paths_env=None,
    num_generations=8,
    do_sample=True,
    temperature=1.0,
    top_p=1.0,
    max_new_tokens=2048,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model, processor = load_model(model_path, device_map)

    dataset_entries = resolve_dataset_entries(test_datasets, data_root, data_paths_env)

    for ds_name, ds_path in dataset_entries:
        print(f"Processing {ds_name}...")
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
            q_text = question_template.format(Question=q_text)
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": q_text}
                    ]
                }
            ])

        outputs_by_example = [[] for _ in data]
        for i in tqdm(range(0, len(messages), batch_size)):
            batch_messages = messages[i:i + batch_size]
            batch_indices = list(range(i, i + len(batch_messages)))
            text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
            image_inputs, video_inputs = process_vision_info(batch_messages)
            print(text)
            inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = repeat_batch_encoding(inputs, num_generations)
            inputs = inputs.to(device_map)
            repeated_indices = repeat_indices(batch_indices, num_generations)
            generation_kwargs = {
                "use_cache": True,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
            }
            generated_ids = model.generate(**inputs, **generation_kwargs)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            batch_output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for idx, output_text in enumerate(batch_output_text):
                input_height = int(inputs["image_grid_thw"][idx][1].item() * 14)
                input_width = int(inputs["image_grid_thw"][idx][2].item() * 14)
                example_idx = repeated_indices[idx]
                outputs_by_example[example_idx].append({
                    "text": output_text,
                    "input_height": input_height,
                    "input_width": input_width,
                })

        results = []
        total_iou = 0.0
        correct_number = 0
        for ex, outs in zip(data, outputs_by_example):
            image_path = os.path.join(image_root, ex["image"])
            image = Image.open(image_path)
            image_width, image_height = image.size

            seg_mask_path = resolve_seg_mask_path(ex.get("seg_mask_path"))
            gt_mask = load_gt_mask(seg_mask_path)

            if not outs:
                outs = [{"text": "", "input_height": 1.0, "input_width": 1.0}]

            for gen_idx, out in enumerate(outs):
                pred_bbox = extract_bbox_answer(out["text"]) or [0.0, 0.0, 0.0, 0.0]
                resized_bbox = resize_bbox(pred_bbox, out["input_height"], out["input_width"], image_height, image_width)

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
                    "generation_index": gen_idx,
                })

        mean_iou = total_iou / len(results) if results else 0.0
        accuracy = correct_number / len(results) * 100 if results else 0.0
        print(f"\nMean IoU of {ds_name}: {mean_iou:.4f}")
        print(f"Accuracy of {ds_name}: {accuracy:.2f}%")

        result_path = os.path.join(output_dir, f"{os.path.basename(model_path)}", f"{ds_name}_seg_r1.json")
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
    model_path = os.getenv("SEG_EVAL_MODEL_PATH", "/workspace/VLM-R1/checkpoints/rl/Qwen2.5-VL-3B-Instruct-segment/checkpoint-900")
    data_root = os.getenv("SEG_EVAL_DATA_ROOT", "/workspace/VLM-R1/data")
    datasets_env = os.getenv("SEG_EVAL_DATASETS", "val")
    test_datasets = [ds.strip() for ds in datasets_env.split(",") if ds.strip()]
    image_root = os.getenv("SEG_EVAL_IMAGE_ROOT", "/workspace/VLM-R1/data")
    output_dir = os.getenv("SEG_EVAL_OUTPUT_DIR", "logs")
    device_map = os.getenv("SEG_EVAL_DEVICE_MAP", "cuda:0")
    batch_size = int(os.getenv("SEG_EVAL_BSZ", "32"))
    sample_num = int(os.getenv("SEG_EVAL_NUM_SAMPLES", "500"))
    num_generations = int(os.getenv("SEG_EVAL_NUM_GENERATIONS", "8"))
    do_sample = strtobool_env(os.getenv("SEG_EVAL_DO_SAMPLE", "true"))
    temperature = float(os.getenv("SEG_EVAL_TEMPERATURE", "1.0"))
    top_p = float(os.getenv("SEG_EVAL_TOP_P", "1.0"))
    max_new_tokens = int(os.getenv("SEG_EVAL_MAX_NEW_TOKENS", "2048"))
    question_template = os.getenv(
        "SEG_EVAL_QUESTION_TEMPLATE",
        Qwen2VLModule.get_question_template(task_type="segment"),
    )
    data_paths_env = os.getenv("SEG_EVAL_DATA_PATHS")
    eval_seg_r1(
        model_path,
        test_datasets,
        data_root,
        image_root,
        question_template,
        output_dir,
        batch_size=batch_size,
        sample_num=sample_num,
        device_map=device_map,
        data_paths_env=data_paths_env,
        num_generations=num_generations,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
