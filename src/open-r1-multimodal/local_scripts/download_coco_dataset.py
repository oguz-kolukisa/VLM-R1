import os
import argparse
import requests
import zipfile
import json
from pycocotools.coco import COCO
from tqdm import tqdm

# Official COCO download links. These contain the raw images and
# annotation files with polygon segmentations which we convert to RLE
# format for training.
TRAIN_URL = "http://images.cocodataset.org/zips/train2014.zip"
VAL_URL = "http://images.cocodataset.org/zips/val2014.zip"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

def download_and_extract(url: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    fname = os.path.join(dest_dir, os.path.basename(url))
    if not os.path.exists(fname):
        print(f"Downloading {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    print(f"Extracting {fname}...")
    with zipfile.ZipFile(fname, "r") as z:
        z.extractall(dest_dir)
    #os.remove(fname)


def convert_split(ann_file: str, img_dir: str, out_path: str):
    """Convert COCO annotations to jsonl format used by ``grpo_jsonl.py``.

    Each line in ``out_path`` contains an object with ``id``, ``image`` and
    ``conversations`` fields so that the dataset can be streamed during
    training.
    """

    coco = COCO(ann_file)
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    with open(out_path, "w") as f:
        for idx, ann_id in enumerate(
            tqdm(coco.getAnnIds(), desc=f"Formatting {os.path.basename(ann_file)}"),
            start=1,
        ):
            ann = coco.loadAnns(ann_id)[0]
            img = coco.loadImgs(ann["image_id"])[0]
            solution = {
                "polygons": ann["segmentation"],
                "size": [img["height"], img["width"]],
            }
            example = {
                "id": idx,
                "image": os.path.join(img_dir, img["file_name"]),
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>Segment the {cats[ann['category_id']]}.",
                    },
                    {
                        "from": "gpt",
                        "value": "<answer>"
                        + json.dumps(solution, separators=(",", ":"))
                        + "</answer>",
                    },
                ],
            }
            f.write(json.dumps(example) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download and format COCO dataset")
    parser.add_argument("--output_dir", default="datasets/coco", help="Directory to store the dataset")
    args = parser.parse_args()

    out = args.output_dir
    download_and_extract(TRAIN_URL, out)
    download_and_extract(VAL_URL, out)
    download_and_extract(ANN_URL, out)

    train_ann = os.path.join(out, "annotations", "instances_train2014.json")
    val_ann = os.path.join(out, "annotations", "instances_val2014.json")
    if os.path.exists(train_ann):
        convert_split(train_ann, "train2014", os.path.join(out, "train.jsonl"))
    if os.path.exists(val_ann):
        convert_split(val_ann, "val2014", os.path.join(out, "val.jsonl"))
    print(f"COCO dataset is ready at {out}")


if __name__ == "__main__":
    main()
