from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from copy import deepcopy
from open_r1.vlm_modules.vlm_module import VLMBaseModule
from PIL import Image

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        additional_output = None
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
            additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs, additional_output
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case "segment":
                return "{Question} First, write your reasoning inside <think>...</think> tags. Next, write the final answer inside <answer>...</answer> tags. The content of <answer> MUST be a valid JSON list of four numbers [x1, y1, x2, y2] in image coordinates. Use exactly that list and no other keys or text."
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    @staticmethod
    def format_reward_segment(completions, **kwargs):
        """
        Reward 1·0 if a completion contains:

            <answer>[x1, y1, x2, y2]</answer>

        where the answer content is valid JSON and parses into a list of four
        numeric values. Anything else earns 0·0.
        """
        import re
        import os
        import json
        from datetime import datetime

        # -- regex helpers -------------------------------------------------------
        ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

        # -- core checker --------------------------------------------------------
        def valid_bbox(text: str) -> bool:
            ans_match = ANSWER_RE.search(text)
            if not ans_match:
                return False
            try:
                obj = json.loads(ans_match.group(1).strip())
            except json.JSONDecodeError:
                return False
            if not isinstance(obj, list) or len(obj) != 4:
                return False
            return all(isinstance(x, (int, float)) for x in obj)

        matches = [valid_bbox(c[0]["content"]) for c in completions]

        # -- optional debugging (unchanged) --------------------------------------
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip([c[0]['content'] for c in completions], matches):
                    f.write(f"Content: {content}\nHas correct format: {match}\n")

        # -- reward vector -------------------------------------------------------
        return [1.0 if match else 0.0 for match in matches]

    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth mask."""
        import re
        import os
        from datetime import datetime
        import json
        import numpy as np
        def mask_iou(pred_mask, gt_mask):
            pred = pred_mask.astype(bool)
            gt = gt_mask.astype(bool)
            inter = float(np.logical_and(pred, gt).sum())
            union = float(np.logical_or(pred, gt).sum())
            return inter / union if union else 0.0

        def load_gt_mask(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
            return mask > 0

        def get_sam2_predictor():
            if not hasattr(get_sam2_predictor, "_predictor"):
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                ckpt = "/workspace/vlm-r1/VLM-R1/sam2/checkpoints/sam2.1_hiera_large.pt"
                if not ckpt:
                    raise ValueError("SAM2_CKPT must be set to the SAM2 checkpoint path.")
                model_cfg = "sam2.1_hiera_l.yaml"
                device = os.getenv("SAM2_DEVICE", "cuda")
                sam2_model = build_sam2(model_cfg, ckpt, device=device)
                get_sam2_predictor._predictor = SAM2ImagePredictor(sam2_model)
            return get_sam2_predictor._predictor

        def resize_bbox(bbox, input_height, input_width, image_height, image_width):
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
            bbox[2] = bbox[2] / input_width * image_width
            bbox[3] = bbox[3] / input_height * image_height
            return bbox
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

        for i, (content, sol) in enumerate(zip(contents, solution)):
            image_grid_thw = kwargs.get("image_grid_thw")[i]
            image_path = kwargs.get("image_path")[i][0]
            image = Image.open(image_path)
            image_width, image_height = image.size
            input_height = int(image_grid_thw[1]*14)
            input_width = int(image_grid_thw[2]*14)

            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)
                        seg_mask_path = kwargs.get("seg_mask_path")
                        if not seg_mask_path:
                            raise ValueError("seg_mask_path is required for mask IoU reward.")
                        predictor = get_sam2_predictor()
                        np_image = np.array(image.convert("RGB"))
                        predictor.set_image(np_image)
                        masks, _, _ = predictor.predict(
                            box=np.array(bbox, dtype=np.float32)[None, :],
                            multimask_output=False,
                        )
                        pred_mask = masks[0].astype(bool)
                        gt_mask = load_gt_mask(seg_mask_path[i])
                        reward = mask_iou(pred_mask, gt_mask)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[i]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards

    
    @staticmethod
    def mask_iou_reward(completions, solution, **kwargs):
        """
        Compute an IoU‑based reward between a predicted bounding box (decoded via
        SAM2) and a ground‑truth segmentation mask.

        **Safe version** — _never raises_. Any failure (e.g. malformed JSON, missing
        files, unavailable dependencies) yields a reward of **0.0** for that
        particular completion. The function therefore always returns a list of
        length `len(completions)` and will _never_ propagate an exception to the
        caller.

        Expected JSON inside `<answer> … </answer>`:
            [x1, y1, x2, y2]
        """
        n = len(completions)
        rewards = [0.0] * n

        try:
            import re
            import json
            import os
            import numpy as np
            from datetime import datetime
        except Exception:
            return rewards

        def mask_iou(pred_mask, gt_mask):
            pred = pred_mask.astype(bool)
            gt = gt_mask.astype(bool)
            inter = float(np.logical_and(pred, gt).sum())
            union = float(np.logical_or(pred, gt).sum())
            return inter / union if union else 0.0

        def load_gt_mask(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
            return mask > 0

        def get_sam2_predictor():
            if not hasattr(get_sam2_predictor, "_predictor"):
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                ckpt = "/workspace/vlm-r1/VLM-R1/sam2/checkpoints/sam2.1_hiera_large.pt"
                if not ckpt:
                    raise ValueError("SAM2_CKPT must be set to the SAM2 checkpoint path.")
                model_cfg = "sam2.1_hiera_l.yaml"
                device = os.getenv("SAM2_DEVICE", "cuda")
                sam2_model = build_sam2(model_cfg, ckpt, device=device)
                get_sam2_predictor._predictor = SAM2ImagePredictor(sam2_model)
            return get_sam2_predictor._predictor

        def resize_bbox(bbox, input_height, input_width, image_height, image_width):
            return [
                bbox[0] / input_width * image_width,
                bbox[1] / input_height * image_height,
                bbox[2] / input_width * image_width,
                bbox[3] / input_height * image_height,
            ]

        def clamp_bbox(bbox, image_height, image_width):
            x1, y1, x2, y2 = bbox
            x1 = min(max(x1, 0.0), image_width - 1)
            x2 = min(max(x2, 0.0), image_width - 1)
            y1 = min(max(y1, 0.0), image_height - 1)
            y2 = min(max(y2, 0.0), image_height - 1)
            return [x1, y1, x2, y2]

        answer_tag_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        debug = os.getenv("DEBUG_MODE", "false").lower() == "true"
        log_path = os.getenv("LOG_PATH", "mask_iou_reward.log")

        for i in range(n):
            try:
                content = completions[i][0]["content"]
                image_grid_thw = kwargs.get("image_grid_thw")[i]
                image_path = kwargs.get("image_path")[i][0]
                image = Image.open(image_path)
                image_width, image_height = image.size
                input_height = int(image_grid_thw[1] * 14)
                input_width = int(image_grid_thw[2] * 14)

                match = answer_tag_pattern.search(content)
                if not match:
                    raise ValueError("No <answer> tag in completion")
                bbox = json.loads(match.group(1).strip())
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError("Predicted answer must be a JSON list of four numbers")
                if not all(isinstance(x, (int, float)) for x in bbox):
                    raise ValueError("Predicted bbox values must be numeric")

                bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)
                bbox = clamp_bbox(bbox, image_height, image_width)

                seg_mask_path = kwargs.get("seg_mask_path")
                if not seg_mask_path:
                    raise ValueError("seg_mask_path is required for mask IoU reward.")

                predictor = get_sam2_predictor()
                np_image = np.array(image.convert("RGB"))
                predictor.set_image(np_image)
                masks, _, _ = predictor.predict(
                    box=np.array(bbox, dtype=np.float32)[None, :],
                    multimask_output=False,
                )
                pred_mask = masks[0].astype(bool)
                gt_mask = load_gt_mask(seg_mask_path[i])
                rewards[i] = mask_iou(pred_mask, gt_mask)

                if debug:
                    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"------------- {current_time} Mask IoU reward -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Predicted bbox: {bbox}\n")
                        f.write(f"Reward: {rewards[i]}\n")

            except Exception as e:
                if debug:
                    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"------------- {current_time} Mask IoU reward error -------------\n")
                        f.write(f"Entry #{i} error: {e}\n\n")
                continue

        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                
                case "rec":
                    return Qwen2VLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported task type: {func}")
        elif func == "mask_iou":
            match task_type:
                case "segment":
                    return Qwen2VLModule.mask_iou_reward
                case _:
                    raise ValueError(f"Unsupported task type: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case "segment":
                    return Qwen2VLModule.format_reward_segment
                case _:
                    raise ValueError(f"Unsupported task type: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")
