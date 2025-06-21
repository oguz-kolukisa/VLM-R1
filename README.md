# VLM-R1: A stable and generalizable R1-style Large Vision-Language Model


## 🛠️ Setup

```bash
conda create -n vlm-r1 python=3.10
conda activate vlm-r1
bash setup.sh
```

## 💪🏻 Training

### Referring Expression Comprehension (REC)

#### 📚 GRPO

1. Run `python src/open-r1-multimodal/local_scripts/download_coco_dataset.py --output_dir <data_dir>` to download the COCO images and convert each instance mask to an 8‑term Chebyshev polynomial representation. The converted annotations will be saved as `<data_dir>/train.jsonl` and `<data_dir>/val.jsonl`, with each line containing an `id`, `image`, and `conversations` field as described below.
2. (Optional) Preview the polygon annotations by running `python src/eval/draw_polygons.py <data_dir>/val.jsonl --image-root <data_dir> --output vis` to generate images with the masks drawn.
3. Change the `data_paths` and `image_folders` in the [run_scripts/run_grpo_rec.sh](run_scripts/run_grpo_rec.sh) file to point to `<data_dir>`.

```bash
# These jsonl files are included in the annotation files at step 2.
# Note: please use jsonl files instead of json files.
data_paths="path/to/refcoco_train.jsonl:path/to/refcocop_train.jsonl:path/to/refcocog_train.jsonl"
image_folders="path/to/coco:path/to/coco:path/to/coco"
```

4. ``bash run_scripts/run_grpo_seg.sh``

> [!NOTE]
> If you encounter 'CUDA out of memory' error, you can try to reduce the `per_device_train_batch_size`.

### For your own data

<div style="text-align: justify;">

We support data loading the jsonl data of this format in [`src/open-r1-multimodal/src/open_r1/grpo_jsonl.py`](src/open-r1-multimodal/src/open_r1/grpo_jsonl.py). Please note that you may need to use different reward functions for your specialized tasks. Welcome to PR to add your own reward functions or share any other interesting findings!

</div>

The jsonl has the format as follows:

```json
{
  "id": 1,
  "image": "Clevr_CoGenT_TrainA_R1/data/images/CLEVR_trainA_000001_16885.png",
  "conversations": [
    {"from": "human", "value": "<image>What number of purple metallic balls are there?"},
    {"from": "gpt", "value": "0"}
  ]
}
```

If you want to use multi-image input, you can use the following format:

```json
{
  "id": 1,
  "image": ["Clevr_CoGenT_TrainA_R1/data/images/CLEVR_trainA_000001_16885.png", "Clevr_CoGenT_TrainA_R1/data/images/CLEVR_trainA_000001_16886.png"],
  "conversations": [
    {"from": "human", "value": "<image><image>What number of purple metallic balls in total within the two images?"},
    {"from": "gpt", "value": "3"}
  ]
}
```

> [!NOTE]
> The image path in the jsonl file should be relative to the image folder specified in `--image_folders`. The absolute path of the input image is constructed as `os.path.join(image_folder, data['image'])`. For example:

- If your jsonl has `"image": "folder1/image1.jpg"`
- And you specify `--image_folders "/path/to/images/"`
- The full image path will be `/path/to/images/folder1/image1.jpg`

Multiple data files and image folders can be specified using ":" as a separator:

```bash
--data_file_paths /path/to/data1.jsonl:/path/to/data2.jsonl \
--image_folders /path/to/images1/:/path/to/images2/
```

The script can be run like this:

```bash
# You could refer to the run_grpo_rec.sh for the example
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
  src/open_r1/grpo_jsonl.py \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json \
    --data_file_paths /path/to/your/data.jsonl \ # can be multiple, separated by ":"
    --image_folders /path/to/your/image/folder \ # can be multiple, separated by ":"
    ...
```

<div style="text-align: justify;">

### Multi-image Input
We provide an example of multi-image script [run_grpo_gui.sh](src/open-r1-multimodal/run_scripts/run_grpo_gui.sh). This task requires the model to analyze two GUI screenshots, taken before and after a user action, to determine if any UI interaction defects are present, which is from [GUI-Testing-Arena](https://huggingface.co/datasets/songjah/GTArena-UI-Defects). Download the [image](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/gui_multi-image.zip) and unzip it into the `/path/to/images/`. Then modify the `image_folders` parameter in the script and run it.

```bash
bash run_scripts/run_grpo_gui.sh
```

</div>

