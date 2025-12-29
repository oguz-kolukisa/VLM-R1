#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_HOME}"

# --- User editable paths ----------------------------------------------------
DATA_DIR=${DATA_DIR:-"${REPO_HOME}/data"}
GREFCOCO_ROOT=${GREFCOCO_ROOT:-"${DATA_DIR}/grefcoco"}
COCO_ROOT=${COCO_ROOT:-"${DATA_DIR}/coco/train2014"}
OUTPUT_JSONL=${OUTPUT_JSONL:-"${DATA_DIR}/grefcoco_train.jsonl"}
MASK_DIR=${MASK_DIR:-"${DATA_DIR}/grefcoco_masks"}
DATASET=${DATASET:-"grefcoco"}
SPLIT_BY=${SPLIT_BY:-"unc"}
SPLIT=${SPLIT:-"train"}
MAX_SAMPLES=${MAX_SAMPLES:-""}           # set to integer for debugging subset
DOWNLOAD=${DOWNLOAD:-"false"}            # set to "true" to allow auto-download
GREFCOCO_URL=${GREFCOCO_URL:-""}         # required when DOWNLOAD=true
COCO_URL=${COCO_URL:-""}                 # required when DOWNLOAD=true
# ----------------------------------------------------------------------------

ARGS=(
    --grefcoco-root "${GREFCOCO_ROOT}"
    --coco-root "${COCO_ROOT}"
    --output-jsonl "${OUTPUT_JSONL}"
    --mask-dir "${MASK_DIR}"
    --dataset "${DATASET}"
    --split-by "${SPLIT_BY}"
    --split "${SPLIT}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
    ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ "${DOWNLOAD}" == "true" ]]; then
    if [[ -z "${GREFCOCO_URL}" || -z "${COCO_URL}" ]]; then
        echo "Set GREFCOCO_URL and COCO_URL when DOWNLOAD=true." >&2
        exit 1
    fi
    ARGS+=(--download --grefcoco-url "${GREFCOCO_URL}" --coco-url "${COCO_URL}")
fi

python src/open-r1-multimodal/local_scripts/prepare_grefcoco_grpo.py "${ARGS[@]}"

echo "gRefCOCO JSONL written to ${OUTPUT_JSONL}"
echo "Segmentation masks stored in ${MASK_DIR}"
