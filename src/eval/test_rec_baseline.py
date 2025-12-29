from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
from PIL import Image
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    
    dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    print(f"Process {rank}/{world_size} initialized on cuda:{local_rank}")
    return local_rank, world_size, rank

local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"

steps = 100
MODEL_PATH=f"/data10/shz/project/LLaMA-Factory/saves/qwen2_5_vl-3b/full/sft/checkpoint-{steps}" 
OUTPUT_PATH="./logs/rec_results_{DATASET}_qwen2_5vl_3b_instruct_sft_{STEPS}.json"

# MODEL_PATH = "/data10/shz/ckpt/vlm-r1-related/Qwen2.5-VL-3B-Instruct"
# OUTPUT_PATH = "./logs/rec_results_{DATASET}_qwen2_5vl_3b_instruct_baseline_{STEPS}.json"

BSZ=4
DATA_ROOT = "/data10/shz/dataset/rec/rec_jsons_processed"

TEST_DATASETS = ['refcoco_val', 'refcocop_val', 'refcocog_val']
IMAGE_ROOT = "/data10/shz/dataset/coco"
SEG_MASK_ROOT = os.getenv("SEG_MASK_ROOT")

# TEST_DATASETS = ['lisa_test']
# IMAGE_ROOT = "/data10/shz/dataset/lisa"

#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": local_rank}, 
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

def extract_bbox_answer(content):
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    # bbox_pattern = r'\[(-?\d*\.?\d+),\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+)\]'
    bbox_match = re.search(bbox_pattern, content)

    if bbox_match:
        bbox = [float(bbox_match.group(1)), float(bbox_match.group(2)), float(bbox_match.group(3)), float(bbox_match.group(4))]
        return bbox
    return [0, 0, 0, 0]

def resize_bbox(bbox, input_height, input_width, image_height, image_width):
    bbox[0] = bbox[0] / input_width * image_width
    bbox[1] = bbox[1] / input_height * image_height
    bbox[2] = bbox[2] / input_width * image_width
    bbox[3] = bbox[3] / input_height * image_height
    return bbox


_SAM2_PREDICTOR = None


def get_sam2_predictor():
    global _SAM2_PREDICTOR
    if _SAM2_PREDICTOR is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        ckpt = os.getenv("SAM2_CKPT")
        if not ckpt:
            raise ValueError("SAM2_CKPT must be set for SAM-based evaluation.")
        model_cfg = os.getenv("SAM2_MODEL", "sam2_hiera_large")
        device = os.getenv("SAM2_DEVICE", "cuda")
        sam2_model = build_sam2(model_cfg, ckpt, device=device)
        _SAM2_PREDICTOR = SAM2ImagePredictor(sam2_model)
    return _SAM2_PREDICTOR


def resolve_seg_mask_path(mask_path):
    if mask_path is None:
        raise ValueError("seg_mask_path missing in dataset item for SAM-based evaluation.")
    if os.path.isabs(mask_path):
        return mask_path
    if SEG_MASK_ROOT is None:
        raise ValueError("SEG_MASK_ROOT environment variable must be set when seg_mask_path is relative.")
    return os.path.join(SEG_MASK_ROOT, mask_path)


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


num_samples = 2000
for ds in TEST_DATASETS:
    if rank == 0:
        print(f"Processing {ds}...")

    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    data = json.load(open(ds_path, "r"))

    random.seed(42)
    random.shuffle(data)
    data = data[:num_samples]
    # QUESTION_TEMPLATE = "{Question}" if steps > 0 else "{Question} Please provide the bounding box coordinate in JSON format."
    QUESTION_TEMPLATE = "{Question} Please provide the bounding box coordinate in JSON format."
    
    # Split data for distributed evaluation
    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]
    
    messages = []

    for x in rank_data:
        image_path = os.path.join(IMAGE_ROOT, x['image'])
        message = [
            # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"file://{image_path}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=x['problem'])
                }
            ]
        }]
        messages.append(message)

    rank_outputs = [] # List to store answers for this rank
    all_outputs = []  # List to store all answers

    # Process data
    for i in tqdm(range(0, len(messages), BSZ), disable=rank != 0):
        batch_messages = messages[i:i + BSZ]
    
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        batch_output = []
        for i, output_text in enumerate(batch_output_text):
            input_height = int(inputs['image_grid_thw'][i][1]*14)
            input_width = int(inputs['image_grid_thw'][i][2]*14)
            image = Image.open(batch_messages[i][0]['content'][0]['image'].split("file://")[1])
            image_width, image_height = image.size
            batch_output.append((output_text, input_height, input_width, image_height, image_width))
            
        rank_outputs.extend(batch_output)

    print(f"Rank {rank} has finished processing {len(rank_outputs)} examples")

    # Gather all outputs from all ranks
    all_outputs = [None] * len(data)
    rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs)]

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, rank_results)
    
    assert gathered_results[-1][-1][0] == len(data) - 1

    # The main process will collect all results
    if rank == 0:
        for results in gathered_results:
            for idx, output in results:
                assert idx < len(all_outputs)
                all_outputs[idx] = output
        assert all_outputs[-1] is not None

        final_output = []
        correct_number = 0

        for input_example, model_output in zip(data, all_outputs):
            original_output, input_height, input_width, image_height, image_width = model_output
            image_path = os.path.join(IMAGE_ROOT, input_example['image'])
            ground_truth_bbox = input_example['solution']
            seg_mask_path = resolve_seg_mask_path(input_example.get('seg_mask_path'))
            model_answer = extract_bbox_answer(original_output)
            resized_model_answer = resize_bbox(model_answer, input_height, input_width, image_height, image_width)

            gt_mask = load_gt_mask(seg_mask_path)
            mask_score = 0.0
            correct = 0
            if model_answer is not None:
                try:
                    pred_mask = sam_mask_from_bbox(image_path, resized_model_answer)
                    mask_score = mask_iou(pred_mask, gt_mask)
                    if mask_score > 0.5:
                        correct = 1
                except Exception as exc:
                    if rank == 0:
                        print(f"SAM inference failed for {image_path}: {exc}")
            correct_number += correct

            result = {
                'image': input_example['image'],
                'question': input_example['problem'],
                'ground_truth_bbox': ground_truth_bbox,
                'seg_mask_path': seg_mask_path,
                'model_output': original_output,
                'extracted_answer': resized_model_answer,
                'mask_iou': mask_score,
                'correct': correct
            }
            final_output.append(result)

        # Calculate and print accuracy
        accuracy = correct_number / len(data) * 100
        print(f"\nAccuracy of {ds}: {accuracy:.2f}%")

        # Save results to a JSON file
        output_path = OUTPUT_PATH.format(DATASET=ds, STEPS=steps)
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, "w") as f:
            json.dump({
                'accuracy': accuracy,
                'results': final_output
            }, f, indent=2)

        print(f"Results saved to {output_path}")
        print("-"*100)
    
    # Synchronize all processes
    dist.barrier()





