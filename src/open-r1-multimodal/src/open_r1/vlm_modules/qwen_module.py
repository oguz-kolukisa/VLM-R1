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
                return "{Question} First, write your reasoning inside <think>...</think> tags. Next, write the final answer inside <answer>...</answer> tags.The content of <answer> MUST be valid JSON of the form <answer>[x1, y1, x2, y2]</answer> Use exactly those and no others."
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
        Reward 1·0 if a completion contains

            <answer> … {"centre":[cx,cy],"coeffs":[…]} … </answer>

        with the structural rules above.  Anything else earns 0·0.
        """
        import re
        import os
        import json
        from datetime import datetime

        # -- regex helpers -------------------------------------------------------
        ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        JSON_RE   = re.compile(r"\{.*?\}", re.DOTALL)        # first {...} inside <answer>

        # -- core checker --------------------------------------------------------
        def valid_polycoords(text: str) -> bool:
            ans_match = ANSWER_RE.search(text)
            if not ans_match:
                return False

            json_match = JSON_RE.search(ans_match.group(1))
            if not json_match:
                return False
            try:
                obj = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return False

            centre = obj.get("centre")
            coeffs = obj.get("coeffs")

            if (not isinstance(centre, list) or len(centre) != 2 or
                not all(isinstance(x, (int, float)) for x in centre)):
                return False
            if (not isinstance(coeffs, list) or len(coeffs) == 0 or
                not all(isinstance(x, (int, float)) for x in coeffs)):
                return False
            return True

        matches = [valid_polycoords(c[0]["content"]) for c in completions]

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

                ckpt = os.getenv("SAM2_CKPT")
                if not ckpt:
                    raise ValueError("SAM2_CKPT must be set to the SAM2 checkpoint path.")
                model_cfg = os.getenv("SAM2_MODEL", "sam2_hiera_large")
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
        Compute an IoU‑based reward between predicted Chebyshev‑polynomial shapes
        and ground‑truth shapes.

        **Safe version** — _never raises_.  Any failure (e.g. malformed JSON, missing
        keys, out‑of‑range indices, unavailable dependencies) yields a reward of
        **0.0** for that particular completion.  The function therefore always
        returns a list of length `len(completions)` and will _never_ propagate an
        exception to the caller.

        Expected JSON inside `<answer> … </answer>`:
            {"centre": [cx, cy], "coeffs": [c0, c1, …], "size": [H, W]}
        The ground truth may optionally include legacy keys `polygon`/`polygons`.
        """

        # ------------------------------------------------------------------ #
        # Early‑exit defaults and best‑effort imports
        # ------------------------------------------------------------------ #
        n = len(completions)
        rewards = [0.0] * n  # ← assume failure everywhere, then overwrite per‑item

        # Heavy deps may be unavailable in the runtime; if so, keep all‑zero reward
        try:
            import re
            import json
            import os
            import numpy as np
            from datetime import datetime
            from numpy.polynomial.chebyshev import chebval
            from pycocotools import mask as maskUtils
        except Exception:
            return rewards  # cannot evaluate IoU without deps → all zeros

        # ------------------------------------------------------------------ #
        # Helpers
        # ------------------------------------------------------------------ #
        ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

        def cheby_to_polygon(centre, coeffs, n_points: int = 72):
            """Decode radial Chebyshev into a flat COCO polygon list [x1, y1, …]."""
            cx, cy = centre
            thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False, dtype=np.float32)
            radii = chebval(np.cos(thetas), coeffs)
            radii = np.maximum(radii, 0.0)  # ensure non‑negative radius
            xs = cx + radii * np.cos(thetas)
            ys = cy + radii * np.sin(thetas)
            return np.stack([xs, ys], axis=1).reshape(-1).tolist()

        def _clip(poly, h, w):
            """Clip polygon coordinates to `[0, w−1] × [0, h−1]` to appease COCO API."""
            xs = poly[0::2]
            ys = poly[1::2]
            xs = [min(max(float(x), 0.0), w - 1) for x in xs]
            ys = [min(max(float(y), 0.0), h - 1) for y in ys]
            clipped = []
            for x, y in zip(xs, ys):
                clipped += [x, y]
            return clipped

        debug = os.getenv("DEBUG_MODE", "false").lower() == "true"
        log_path = os.getenv("LOG_PATH", "mask_iou_reward.log")

        # ------------------------------------------------------------------ #
        # Main evaluation loop — protect _every_ iteration individually
        # ------------------------------------------------------------------ #
        for idx in range(n):
            try:
                # ------------------ fetch content & GT safely ------------- #
                try:
                    content = completions[idx][0]["content"]
                except Exception:
                    content = ""  # triggers reward = 0 below

                try:
                    sol_raw = solution[idx]
                except Exception:
                    sol_raw = ""  # missing GT → reward = 0

                # ------------------ parse ground truth ------------------- #
                try:
                    sol_json_strs = ANSWER_TAG.findall(sol_raw)
                    if not sol_json_strs:
                        raise ValueError("No <answer> tag in GT")
                    gt_obj = json.loads(sol_json_strs[-1].strip())
                except Exception:
                    raise  # handled by outer except → reward = 0

                # GT image size (required)
                h, w = (gt_obj.get("size") or gt_obj.get("sizeHW") or [None, None])
                if h is None or w is None:
                    raise ValueError("Missing size in ground‑truth object")

                # Ground‑truth polygon (Chebyshev or legacy polygon list)
                if "centre" in gt_obj and "coeffs" in gt_obj:
                    gt_poly = cheby_to_polygon(gt_obj["centre"], gt_obj["coeffs"])
                else:
                    gt_poly = gt_obj.get("polygon") or gt_obj.get("polygons")
                    if not gt_poly:
                        raise ValueError("No GT polygon information")

                # ------------------ parse prediction --------------------- #
                pred_match = ANSWER_TAG.search(content)
                if not pred_match:
                    raise ValueError("No <answer> tag in completion")

                pred_obj = json.loads(pred_match.group(1).strip())

                if "centre" in pred_obj and "coeffs" in pred_obj:
                    pred_poly = cheby_to_polygon(pred_obj["centre"], pred_obj["coeffs"])
                else:
                    pred_poly = pred_obj.get("polygon") or pred_obj.get("polygons")
                    if not pred_poly:
                        raise ValueError("No predicted polygon information")

                # ------------------ IoU via COCO mask API ---------------- #
                gt_poly_clipped   = _clip(gt_poly,   h, w)
                pred_poly_clipped = _clip(pred_poly, h, w)

                gt_rle   = maskUtils.merge(maskUtils.frPyObjects([gt_poly_clipped],   h, w))
                pred_rle = maskUtils.merge(maskUtils.frPyObjects([pred_poly_clipped], h, w))
                iou = maskUtils.iou([pred_rle], [gt_rle], [False])[0][0]

                rewards[idx] = float(iou) if np.isfinite(iou) else 0.0

            except Exception as e:
                # Any failure → leave rewards[idx] at default 0.0
                if debug:
                    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"------------- {current_time} Mask IoU reward: 0.0000 -------------\n")
                        f.write(f"Entry #{idx} error: {e}\n\n")
                continue  # proceed to next completion safely

        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                
                case "rec":
                    return Qwen2VLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "mask_iou":
            match task_type:
                case "segment":
                    return Qwen2VLModule.mask_iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case "segment":
                    return Qwen2VLModule.format_reward_segment
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")
