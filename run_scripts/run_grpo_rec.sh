#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: ${REPO_HOME}"

# -----------------------------------------------------------------------------
# Configure dataset and model paths. Override via env before calling this script.
# -----------------------------------------------------------------------------
DATA_DIR=${DATA_DIR:-"${REPO_HOME}/data"}
REFCOCO_EXPORT_BASE=${REFCOCO_EXPORT_BASE:-"${DATA_DIR}/refcoco_exports"}
COCO_IMAGE_ROOT=${COCO_IMAGE_ROOT:-"${DATA_DIR}/coco"}
MODEL_PATH=${MODEL_PATH:-"${REPO_HOME}/checkpoints/Qwen2.5-VL-3B-Instruct"}
IS_REWARD_CUSTOMIZED=${IS_REWARD_CUSTOMIZED:-"true"}
EXP_NAME="${1:-${EXP_NAME:-Qwen2.5-VL-3B-Instruct-rec}}"
TASK_TYPE=${TASK_TYPE:-"rec"}
GPU_PER_NODE=${GPU_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"12349"}

if [[ ! -d "${REFCOCO_EXPORT_BASE}" ]]; then
    echo "RefCOCO export directory not found: ${REFCOCO_EXPORT_BASE}"
    echo "Run prepare_refcoco_dataset.sh and export the metadata first."
    exit 1
fi

mapfile -t DATA_PATHS < <(find "${REFCOCO_EXPORT_BASE}" -maxdepth 2 -name "metadata.jsonl" | sort)
if [[ ${#DATA_PATHS[@]} -eq 0 ]]; then
    echo "No metadata.jsonl files detected under ${REFCOCO_EXPORT_BASE}."
    echo "Execute prepare_refcoco_dataset.sh and run the printed python commands to generate exports."
    exit 1
fi

IMAGE_FOLDERS=()
for _ in "${DATA_PATHS[@]}"; do
    IMAGE_FOLDERS+=("${COCO_IMAGE_ROOT}")
done

join_colon() {
    local IFS=':'
    echo "$*"
}

data_paths="$(join_colon "${DATA_PATHS[@]}")"
image_folders="$(join_colon "${IMAGE_FOLDERS[@]}")"

echo "Detected data_paths:"
printf '  %s\n' "${DATA_PATHS[@]}"
echo "Image folders:"
printf '  %s\n' "${IMAGE_FOLDERS[@]}"
echo "Model path: ${MODEL_PATH}"

cd "${REPO_HOME}/src/open-r1-multimodal"

export DEBUG_MODE="${DEBUG_MODE:-true}"
mkdir -p "${REPO_HOME}/runs/${EXP_NAME}/log"
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

torchrun --nproc_per_node="${GPU_PER_NODE}" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir "${REPO_HOME}/checkpoints/rl/${EXP_NAME}" \
    --resume_from_checkpoint True \
    --model_name_or_path "${MODEL_PATH}" \
    --data_file_paths "${data_paths}" \
    --image_folders "${image_folders}" \
    --is_reward_customized_from_vlm_module "${IS_REWARD_CUSTOMIZED}" \
    --task_type "${TASK_TYPE}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-8}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-2}" \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name "${EXP_NAME}" \
    --data_seed 42 \
    --save_steps 100 \
    --num_generations 8 \
    --max_completion_length 2048 \
    --reward_funcs accuracy format \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name this_is_not_used \
    --deepspeed "${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json"

echo "Training completed for ${EXP_NAME}"
