#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_HOME}"

# --- User editable values ---------------------------------------------------
DATA_DIR=${DATA_DIR:-"${REPO_HOME}/data"}
REFCOCO_EXPORT_BASE=${REFCOCO_EXPORT_BASE:-"${DATA_DIR}/refcoco_exports"}
COCO_IMAGE_ROOT=${COCO_IMAGE_ROOT:-"${DATA_DIR}/coco"}
RUN_NAME=${RUN_NAME:-"Qwen2.5-VL-3B-Instruct-seg"}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-100}
MODEL_PATH=${MODEL_PATH:-"${REPO_HOME}/checkpoints/rl/${RUN_NAME}/checkpoint-${CHECKPOINT_STEPS}"}
MODEL_PATH="okolukisa1/Qwen2.5-VL-3B-Instruct-rec-seg"
IMAGE_ROOT=${IMAGE_ROOT:-"${COCO_IMAGE_ROOT}/train2014"}
SEG_MASK_ROOT=${SEG_MASK_ROOT:-"${REFCOCO_EXPORT_BASE}"}
SAM2_CKPT=${SAM2_CKPT:-"${REPO_HOME}/checkpoints/sam2/sam2_hiera_large.pt"}
SAM2_MODEL=${SAM2_MODEL:-"sam2_hiera_large"}
SAM2_DEVICE=${SAM2_DEVICE:-"cuda"}
TEST_DATASETS=${TEST_DATASETS:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_HOME}/logs"}
NUM_GPUS=${NUM_GPUS:-1}
NUM_SAMPLES=${NUM_SAMPLES:-20}
BSZ=${BSZ:-2}
# ----------------------------------------------------------------------------

mkdir -p "${DATA_DIR}"
mkdir -p "${REPO_HOME}/logs"

if [[ ! -f "${SAM2_CKPT}" ]]; then
  echo "SAM2 checkpoint not found at ${SAM2_CKPT}. Set SAM2_CKPT to a valid file." >&2
  exit 1
fi

if [[ ! -d "${REFCOCO_EXPORT_BASE}" ]]; then
  echo "RefCOCO export directory not found at ${REFCOCO_EXPORT_BASE}." >&2
  echo "Run prepare_refcoco_dataset.sh and export the metadata first." >&2
  exit 1
fi

if [[ ! -d "${SEG_MASK_ROOT}" ]]; then
  echo "SEG_MASK_ROOT not found at ${SEG_MASK_ROOT}. Set SEG_MASK_ROOT to a valid directory." >&2
  exit 1
fi

mapfile -t DATA_PATHS < <(find "${REFCOCO_EXPORT_BASE}" -maxdepth 2 -name "metadata.jsonl" | sort)
if [[ ${#DATA_PATHS[@]} -eq 0 ]]; then
  echo "No metadata.jsonl files detected under ${REFCOCO_EXPORT_BASE}." >&2
  echo "Execute prepare_refcoco_dataset.sh and run the printed python commands to generate exports." >&2
  exit 1
fi

FILTERED_DATA_PATHS=()
if [[ -n "${TEST_DATASETS}" ]]; then
  IFS=',' read -r -a REQUESTED_DATASETS <<< "${TEST_DATASETS}"
  for dataset in "${REQUESTED_DATASETS[@]}"; do
    dataset="$(echo "${dataset}" | xargs)"
    [[ -z "${dataset}" ]] && continue
    candidate="${REFCOCO_EXPORT_BASE}/${dataset}/metadata.jsonl"
    if [[ -f "${candidate}" ]]; then
      FILTERED_DATA_PATHS+=("${candidate}")
      continue
    fi
    for path in "${DATA_PATHS[@]}"; do
      if [[ "${path}" == *"/${dataset}/metadata.jsonl" ]]; then
        FILTERED_DATA_PATHS+=("${path}")
      fi
    done
  done
  if [[ ${#FILTERED_DATA_PATHS[@]} -eq 0 ]]; then
    echo "Requested datasets did not match exports under ${REFCOCO_EXPORT_BASE}." >&2
    exit 1
  fi
else
  FILTERED_DATA_PATHS=("${DATA_PATHS[@]}")
fi

join_colon() {
  local IFS=':'
  echo "$*"
}

DATA_PATHS_STR="$(join_colon "${FILTERED_DATA_PATHS[@]}")"

export SEG_MASK_ROOT
export SAM2_CKPT SAM2_MODEL SAM2_DEVICE
export SEG_EVAL_MODEL_PATH="${MODEL_PATH}"
export SEG_EVAL_DATA_ROOT="${REFCOCO_EXPORT_BASE}"
export SEG_EVAL_IMAGE_ROOT="${IMAGE_ROOT}"
export SEG_EVAL_DATA_PATHS="${DATA_PATHS_STR}"
export SEG_EVAL_OUTPUT_DIR="${OUTPUT_DIR}"
export SEG_EVAL_NUM_SAMPLES="${NUM_SAMPLES}"
export SEG_EVAL_BSZ="${BSZ}"

cd "${REPO_HOME}/src/eval"
torchrun --nproc_per_node="${NUM_GPUS}" test_seg_r1.py

echo "Evaluation complete. Results saved under ${OUTPUT_DIR}"
