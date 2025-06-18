import os
import argparse
import requests
import zipfile
import json
from pycocotools.coco import COCO
from tqdm import tqdm

TRAIN_URL = "https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/train2014.zip"
VAL_URL = "https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/val2014.zip"
ANN_URL = "https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/annotations_trainval2014.zip"
REC_JSON_URL = "https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/rec_jsons_processed.zip"

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
    os.remove(fname)


def convert_split(ann_file: str, img_dir: str, out_path: str):
    coco = COCO(ann_file)
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    data = []
    for ann_id in tqdm(coco.getAnnIds(), desc=f"Formatting {os.path.basename(ann_file)}"):
        ann = coco.loadAnns(ann_id)[0]
        img = coco.loadImgs(ann["image_id"])[0]
        rle = coco.annToRLE(ann)
        rle["counts"] = rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]
        example = {
            "image": os.path.join(img_dir, img["file_name"]),
            "problem": f"Segment the {cats[ann['category_id']]}.",
            "normal_caption": cats[ann["category_id"]],
            "solution": rle,
        }
        data.append(example)
    with open(out_path, "w") as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser(description="Download and format COCO dataset")
    parser.add_argument("--output_dir", default="datasets/coco", help="Directory to store the dataset")
    args = parser.parse_args()

    out = args.output_dir
    download_and_extract(TRAIN_URL, out)
    download_and_extract(VAL_URL, out)
    download_and_extract(ANN_URL, out)
    download_and_extract(REC_JSON_URL, out)

    train_ann = os.path.join(out, "annotations", "instances_train2014.json")
    val_ann = os.path.join(out, "annotations", "instances_val2014.json")
    if os.path.exists(train_ann):
        convert_split(train_ann, "train2014", os.path.join(out, "train.json"))
    if os.path.exists(val_ann):
        convert_split(val_ann, "val2014", os.path.join(out, "val.json"))
    print(f"COCO dataset is ready at {out}")


if __name__ == "__main__":
    main()
