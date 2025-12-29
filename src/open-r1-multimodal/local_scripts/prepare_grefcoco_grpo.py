import argparse
import importlib
import json
import os
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class DownloadConfig:
    url: str
    dest_dir: Path
    filename: str


def download_and_extract(config: DownloadConfig) -> Path:
    config.dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = config.dest_dir / config.filename
    if not archive_path.exists():
        print(f"Downloading {config.url} -> {archive_path}")
        urllib.request.urlretrieve(config.url, archive_path)
    else:
        print(f"Using existing archive: {archive_path}")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(config.dest_dir)
    return config.dest_dir


def load_grefer_module(grefcoco_root: Path):
    sys.path.insert(0, str(grefcoco_root))
    return importlib.import_module("grefer")


def build_grefer(grefcoco_root: Path, dataset: str, split_by: str):
    grefer = load_grefer_module(grefcoco_root)
    cls = getattr(grefer, "G_REFER", None) or getattr(grefer, "REFER", None)
    if cls is None:
        raise ValueError("Could not find G_REFER or REFER in grefer.py")
    try:
        return cls(str(grefcoco_root), dataset=dataset, splitBy=split_by)
    except TypeError:
        return cls(str(grefcoco_root), dataset, split_by)


def get_ref_ids(refer, split: str):
    if hasattr(refer, "getRefIds"):
        try:
            return refer.getRefIds(split=split)
        except TypeError:
            return refer.getRefIds(split)
    raise ValueError("grefer object does not provide getRefIds")


def load_ref(refer, ref_id: int):
    if hasattr(refer, "loadRefs"):
        refs = refer.loadRefs(ref_id)
        return refs[0] if isinstance(refs, list) else refs
    raise ValueError("grefer object does not provide loadRefs")


def load_ann(refer, ann_id: int):
    if hasattr(refer, "loadAnns"):
        anns = refer.loadAnns(ann_id)
        return anns[0] if isinstance(anns, list) else anns
    raise ValueError("grefer object does not provide loadAnns")


def load_img(refer, image_id: int):
    if hasattr(refer, "loadImgs"):
        imgs = refer.loadImgs(image_id)
        return imgs[0] if isinstance(imgs, list) else imgs
    raise ValueError("grefer object does not provide loadImgs")


def ann_to_mask(ann, height: int, width: int):
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError("pycocotools is required to decode segmentation masks.") from exc

    segm = ann.get("segmentation")
    if segm is None:
        raise ValueError("Annotation missing segmentation data.")
    if isinstance(segm, list):
        rles = mask_utils.frPyObjects(segm, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segm, dict) and isinstance(segm.get("counts"), list):
        rle = mask_utils.frPyObjects(segm, height, width)
    else:
        rle = segm
    mask = mask_utils.decode(rle)
    return mask


def save_mask(mask: np.ndarray, mask_path: Path) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(mask_uint8).save(mask_path)


def sentence_text(sentence):
    if isinstance(sentence, dict):
        return sentence.get("sent") or sentence.get("raw") or sentence.get("sentence")
    return str(sentence)


def build_conversation(sentence: str, mask_path: str):
    problem = f"{sentence}"
    answer = json.dumps({"mask_path": mask_path})
    return [
        {"from": "human", "value": f"<image> {problem}"},
        {"from": "gpt", "value": f"<answer> {answer} </answer>"},
    ]


def main():
    parser = argparse.ArgumentParser(description="Prepare gRefCOCO JSONL for GRPO training.")
    parser.add_argument("--grefcoco-root", type=Path, required=True, help="Path to gRefCOCO repo/data root.")
    parser.add_argument("--coco-root", type=Path, required=True, help="Path to COCO images root.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--mask-dir", type=Path, required=True, help="Output directory for GT masks.")
    parser.add_argument("--dataset", type=str, default="grefcoco", help="Dataset name for grefer.")
    parser.add_argument("--split-by", type=str, default="unc", help="Split-by for grefer (e.g., unc/umd).")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val/test).")
    parser.add_argument("--download", action="store_true", help="Download gRefCOCO annotations or COCO images.")
    parser.add_argument("--grefcoco-url", type=str, default=None, help="Zip URL for gRefCOCO annotations.")
    parser.add_argument("--coco-url", type=str, default=None, help="Zip URL for COCO images.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples for debugging.")
    args = parser.parse_args()

    if args.download:
        if args.grefcoco_url is None or args.coco_url is None:
            raise ValueError("--grefcoco-url and --coco-url must be set when using --download.")
        download_and_extract(
            DownloadConfig(
                url=args.grefcoco_url,
                dest_dir=args.grefcoco_root,
                filename="grefcoco.zip",
            )
        )
        download_and_extract(
            DownloadConfig(
                url=args.coco_url,
                dest_dir=args.coco_root,
                filename="coco_images.zip",
            )
        )

    refer = build_grefer(args.grefcoco_root, dataset=args.dataset, split_by=args.split_by)
    ref_ids = get_ref_ids(refer, args.split)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    num_written = 0
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for ref_id in ref_ids:
            ref = load_ref(refer, ref_id)
            ann = load_ann(refer, ref["ann_id"])
            img = load_img(refer, ref["image_id"])
            file_name = img.get("file_name")
            if file_name is None:
                raise ValueError("Image record missing file_name.")
            height = img.get("height")
            width = img.get("width")
            mask = ann_to_mask(ann, height, width)

            sentences = ref.get("sentences", [])
            if not sentences:
                sentences = [ref.get("sentence", "")]

            for sent_idx, sent in enumerate(sentences):
                text = sentence_text(sent)
                if not text:
                    continue
                mask_filename = f"{ref_id}_{sent_idx}.png"
                mask_path = args.mask_dir / mask_filename
                save_mask(mask, mask_path)
                item = {
                    "id": f"{ref_id}_{sent_idx}",
                    "image": file_name,
                    "seg_mask_path": mask_filename,
                    "conversations": build_conversation(text, mask_filename),
                }
                f.write(json.dumps(item) + "\n")
                num_written += 1
                if args.max_samples and num_written >= args.max_samples:
                    break
            if args.max_samples and num_written >= args.max_samples:
                break
    print(f"Wrote {num_written} samples to {args.output_jsonl}")


if __name__ == "__main__":
    main()
