#!/usr/bin/env python3
"""Visualize polygon annotations on the corresponding images."""
import os
import json
import argparse

import cv2
import numpy as np
from numpy.polynomial.chebyshev import chebval


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


def draw_on_image(img, polygon, color=(0, 255, 0)):
    """Draw a polygon on the image inplace."""
    pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)


def main():
    parser = argparse.ArgumentParser(description="Draw polygons from a jsonl dataset")
    parser.add_argument("jsonl", help="Path to dataset jsonl file")
    parser.add_argument("--image-root", default="", help="Root directory for image paths")
    parser.add_argument("--output", required=True, help="Directory to write visualizations")
    parser.add_argument("--limit", type=int, default=50, help="Number of samples to draw")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    with open(args.jsonl, "r") as f:
        data = [json.loads(line) for line in f]

    for ex in data[: args.limit]:
        img_path = os.path.join(args.image_root, ex["image"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
        sol = ex.get("solution", {})
        if "centre" in sol and "coeffs" in sol:
            poly = cheby_to_polygon(sol["centre"], sol["coeffs"])
        else:
            poly = sol.get("polygon") or sol.get("polygons")
        if poly:
            polys = poly if isinstance(poly[0], list) else [poly]
            for p in polys:
                draw_on_image(img, p)
        out_file = os.path.join(args.output, f"{ex['id']}.jpg")
        cv2.imwrite(out_file, img)


if __name__ == "__main__":
    main()
