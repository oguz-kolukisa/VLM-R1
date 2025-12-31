#!/usr/bin/env python
import os
import re
import json
import tempfile
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
OPEN_R1_SRC = os.path.join(REPO_ROOT, "src", "open-r1-multimodal", "src")
if OPEN_R1_SRC not in os.sys.path:
    os.sys.path.append(OPEN_R1_SRC)

from open_r1.vlm_modules.qwen_module import Qwen2VLModule  # noqa: E402

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

_MODEL = None
_PROCESSOR = None
_SAM2_PREDICTOR = None


def extract_bbox_answer(content):
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
    bbox_match = re.search(
        r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]",
        answer_text,
    )
    if bbox_match:
        return [float(bbox_match.group(i)) for i in range(1, 5)]
    return None


def resize_bbox(bbox, input_height, input_width, image_height, image_width):
    bbox[0] = bbox[0] / input_width * image_width
    bbox[1] = bbox[1] / input_height * image_height
    bbox[2] = bbox[2] / input_width * image_width
    bbox[3] = bbox[3] / input_height * image_height
    return bbox


def get_model():
    global _MODEL, _PROCESSOR
    if _MODEL is None or _PROCESSOR is None:
        model_path = os.getenv("DEMO_MODEL_PATH", "Qwen/Qwen2.5-VL-3B-Instruct")
        device_map = os.getenv("DEMO_DEVICE_MAP", "cuda:0")
        _MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        )
        _PROCESSOR = AutoProcessor.from_pretrained(model_path)
    return _MODEL, _PROCESSOR


def get_sam2_predictor():
    global _SAM2_PREDICTOR
    if _SAM2_PREDICTOR is None:
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = os.getenv("SAM2_DEVICE", "cuda")
        model_id = os.getenv("SAM2_MODEL_ID", "facebook/sam2.1-hiera-large")
        sam2_model = build_sam2_hf(model_id, device=device)
        _SAM2_PREDICTOR = SAM2ImagePredictor(sam2_model)
    return _SAM2_PREDICTOR


def apply_mask(image: Image.Image, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.4):
    image = image.convert("RGB")
    overlay = np.array(image).copy()
    overlay[mask] = (overlay[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


def predict(image: Image.Image, prompt: str) -> Tuple[Optional[Image.Image], str]:
    if image is None or not prompt:
        return None, "Please provide an image and a prompt."

    model, processor = get_model()
    question_template = os.getenv(
        "DEMO_QUESTION_TEMPLATE",
        Qwen2VLModule.get_question_template(task_type="segment"),
    )
    question = question_template.format(Question=prompt)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device_map = os.getenv("DEMO_DEVICE_MAP", "cuda:0")
    inputs = inputs.to(device_map)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=int(os.getenv("DEMO_MAX_NEW_TOKENS", "2048")),
        do_sample=os.getenv("DEMO_DO_SAMPLE", "true").lower() in ("1", "true", "yes", "on"),
        temperature=float(os.getenv("DEMO_TEMPERATURE", "1.0")),
        top_p=float(os.getenv("DEMO_TOP_P", "1.0")),
    )
    generated_ids_trimmed = [generated_ids[0][len(inputs.input_ids[0]):]]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    input_height = int(inputs["image_grid_thw"][0][1].item() * 14)
    input_width = int(inputs["image_grid_thw"][0][2].item() * 14)
    image_width, image_height = image.size

    bbox = extract_bbox_answer(output_text)
    if bbox is None:
        return image, f"No bbox found in model output. Raw output:\n{output_text}"

    resized_bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)

    predictor = get_sam2_predictor()
    np_image = np.array(image.convert("RGB"))
    predictor.set_image(np_image)
    masks, _, _ = predictor.predict(
        box=np.array(resized_bbox, dtype=np.float32)[None, :],
        multimask_output=False,
    )
    mask = masks[0] > 0
    overlay = apply_mask(image, mask)

    return overlay, f"Model output:\n{output_text}\n\nBBox: {resized_bbox}"


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# Segmentation Demo (Qwen2.5-VL + SAM2)\nUpload an image, enter a prompt, and view the predicted segmentation mask.")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input Image")
                prompt_input = gr.Textbox(lines=2, label="Prompt", placeholder="Describe the object to segment...")
                run_button = gr.Button("Run Segmentation")
            with gr.Column():
                image_output = gr.Image(type="pil", label="Masked Output")
                text_output = gr.Textbox(lines=6, label="Details")

        run_button.click(predict, inputs=[image_input, prompt_input], outputs=[image_output, text_output])
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("DEMO_PORT", "7860")))
