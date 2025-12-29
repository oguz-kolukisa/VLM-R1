#!/usr/bin/env python3
"""
Download COCO 2014 + convert each instance mask
→ 8-term radial Chebyshev polynomial (plus centre).

JSONL schema per line
---------------------
{
  "id": ...,
  "image": "...",
  "conversations": [
      {"from": "human", "value": "<image>What is the segmentation polynomial coordinates of the ..."},
      {"from": "gpt",
       "value": "<answer>{\"centre\":[cx,cy],\"coeffs\":[c0…c7]}</answer>"}
  ]
}
"""

import os
import argparse
import requests
import zipfile
import json
from typing import List

import numpy as np                       # NEW
import cv2                               # NEW
from numpy.polynomial.chebyshev import chebfit  # NEW
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils      # NEW
from tqdm import tqdm

# ------------------------------------------------------------------
# ↓↓↓ unchanged: download helpers ---------------------------------
TRAIN_URL = "http://images.cocodataset.org/zips/train2014.zip"
VAL_URL   = "http://images.cocodataset.org/zips/val2014.zip"
ANN_URL   = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
# ------------------------------------------------------------------


def download_and_extract(url: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    fname = os.path.join(dest_dir, os.path.basename(url))
    if not os.path.exists(fname):
        print(f"Downloading {url} …")
        with requests.get(url, stream=True) as r, open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                f.write(chunk)
    print(f"Extracting {fname} …")
    with zipfile.ZipFile(fname, "r") as z:
        z.extractall(dest_dir)


# ------------------------------------------------------------------
# NEW: helper – convert bitmap mask → (centre, coeffs)
# ------------------------------------------------------------------
def mask_to_cheby(mask: np.ndarray, k: int = 8, n_angles: int = 72):
    """
    Args
    ----
    mask : (H, W) uint8 {0,1} binary mask of ONE instance
    k    : number of Chebyshev coefficients to keep
    n_angles : angular samples (resolution of ground truth fit)

    Returns
    -------
    centre (tuple) : (cx, cy) float pixel coordinates
    coeffs (List[float]) : length-`k` Chebyshev coefficients
    """
    h, w = mask.shape

    # 1. robust centre = distance-transform peak
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(dist.argmax(), dist.shape)
    centre = np.array([cx, cy], dtype=np.float32)

    # 2. sample radial distances
    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False, dtype=np.float32)
    radii: List[float] = []
    max_len = float(max(h, w))

    for th in thetas:
        dx, dy = np.cos(th), np.sin(th)
        lo, hi = 0.0, max_len
        # binary search along ray for mask boundary
        for _ in range(11):  # ~0.05-pixel precision
            mid = (lo + hi) * 0.5
            x, y = centre + mid * np.array([dx, dy])
            if 0 <= int(y) < h and 0 <= int(x) < w and mask[int(y), int(x)]:
                lo = mid
            else:
                hi = mid
        radii.append(lo)
    radii = np.asarray(radii, dtype=np.float32)

    # 3. least-squares Chebyshev fit on x = cos(theta)
    x = np.cos(thetas)
    coeffs = chebfit(x, radii, k - 1)          # shape (k,)

    return (float(cx), float(cy)), coeffs.astype(np.float32).tolist()


# ------------------------------------------------------------------
# CHANGED: convert_split → uses mask_to_cheby
# ------------------------------------------------------------------
def convert_split(ann_file: str, img_dir: str, out_path: str,
                  k: int = 8, n_angles: int = 72):
    """Convert COCO annotations to jsonl with Chebyshev polynomial coords."""
    coco = COCO(ann_file)
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    with open(out_path, "w") as f:
        for idx, ann_id in enumerate(
            tqdm(coco.getAnnIds(), desc=f"Formatting {os.path.basename(ann_file)}"),
            start=1,
        ):
            ann = coco.loadAnns(ann_id)[0]
            img = coco.loadImgs(ann["image_id"])[0]

            # ------------------------------------------------------------------
            # obtain binary mask of this annotation
            rle = coco.annToRLE(ann)                  # handles polygons & RLE
            mask = mask_utils.decode(rle)
            if mask.ndim == 3:                        # union of multiple RLE parts
                mask = np.any(mask, axis=2).astype(np.uint8)
            # ------------------------------------------------------------------
            centre, coeffs = mask_to_cheby(mask, k=k, n_angles=n_angles)

            solution = {
                "centre": [round(c, 2) for c in centre],
                "coeffs": [round(c, 4) for c in coeffs],
                "size":   [img["height"], img["width"]],   # keep file size small
            }

            example = {
                "id": idx,
                "image": os.path.join(img_dir, img["file_name"]),
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"<image>What is the segmentation polynomial "
                            f"coordinates of the {cats[ann['category_id']]}."
                        ),
                    },
                    {
                        "from": "gpt",
                        "value": "<answer>" + json.dumps(solution, separators=(",", ":")) + "</answer>",
                    },
                ],
            }
            f.write(json.dumps(example) + "\n")


# ------------------------------------------------------------------
# ↓↓↓ unchanged: CLI wiring ---------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download and format COCO dataset")
    parser.add_argument("--output_dir", default="datasets/coco", help="Directory to store the dataset")
    parser.add_argument("--order", type=int, default=8, help="Chebyshev polynomial order (K)")
    args = parser.parse_args()

    out = args.output_dir
    download_and_extract(TRAIN_URL, out)
    download_and_extract(VAL_URL, out)
    download_and_extract(ANN_URL, out)

    train_ann = os.path.join(out, "annotations", "instances_train2014.json")
    val_ann   = os.path.join(out, "annotations", "instances_val2014.json")
    if os.path.exists(train_ann):
        convert_split(train_ann, "train2014", os.path.join(out, "train.jsonl"), k=args.order)
    if os.path.exists(val_ann):
        convert_split(val_ann, "val2014", os.path.join(out, "val.jsonl"), k=args.order)
    print(f"COCO dataset is ready at {out}")


if __name__ == "__main__":
    main()
