#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_HOME}"

# --- User editable values ---------------------------------------------------
DATA_DIR=${DATA_DIR:-"${REPO_HOME}/data"}
RUN_NAME=${RUN_NAME:-"Qwen2.5-VL-3B-Instruct-rec"}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-100}
MODEL_PATH=${MODEL_PATH:-"${REPO_HOME}/checkpoints/rl/${RUN_NAME}/checkpoint-${CHECKPOINT_STEPS}"}
DATA_ROOT=${DATA_ROOT:-"${DATA_DIR}/rec_jsons_processed"}
IMAGE_ROOT=${IMAGE_ROOT:-"${DATA_DIR}/coco/train2014"}
SEG_MASK_ROOT=${SEG_MASK_ROOT:-"${DATA_DIR}/grefcoco_masks"}
SAM2_CKPT=${SAM2_CKPT:-"${REPO_HOME}/checkpoints/sam2/sam2_hiera_large.pt"}
SAM2_MODEL=${SAM2_MODEL:-"sam2_hiera_large"}
SAM2_DEVICE=${SAM2_DEVICE:-"cuda"}
TEST_DATASETS=${TEST_DATASETS:-"refcoco_val,refcocop_val,refcocog_val"}
OUTPUT_PATH=${OUTPUT_PATH:-"${REPO_HOME}/logs/rec_results_{DATASET}_${RUN_NAME}_${CHECKPOINT_STEPS}.json"}
NUM_GPUS=${NUM_GPUS:-8}
NUM_SAMPLES=${NUM_SAMPLES:-2000}
BSZ=${BSZ:-2}
# ----------------------------------------------------------------------------

mkdir -p "${DATA_DIR}"
mkdir -p "${REPO_HOME}/logs"

export SEG_MASK_ROOT
export SAM2_CKPT SAM2_MODEL SAM2_DEVICE
export REC_EVAL_STEPS="${CHECKPOINT_STEPS}"
export REC_EVAL_RUN_NAME="${RUN_NAME}"
export REC_EVAL_MODEL_PATH="${MODEL_PATH}"
export REC_EVAL_DATA_ROOT="${DATA_ROOT}"
export REC_EVAL_IMAGE_ROOT="${IMAGE_ROOT}"
export REC_EVAL_DATASETS="${TEST_DATASETS}"
export REC_EVAL_OUTPUT_PATH="${OUTPUT_PATH}"
export REC_EVAL_NUM_SAMPLES="${NUM_SAMPLES}"
export REC_EVAL_BSZ="${BSZ}"

cd "${REPO_HOME}/src/eval"
torchrun --nproc_per_node="${NUM_GPUS}" test_rec_r1.py

echo "Evaluation complete. Results saved under ${OUTPUT_PATH}"
