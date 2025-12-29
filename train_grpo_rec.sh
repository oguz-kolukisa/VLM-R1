#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_HOME}/src/open-r1-multimodal"

# --- User editable paths ----------------------------------------------------
DATA_DIR=${DATA_DIR:-"${REPO_HOME}/data"}
DATA_PATHS=${DATA_PATHS:-"${DATA_DIR}/grefcoco_train.jsonl"}
IMAGE_FOLDERS=${IMAGE_FOLDERS:-"${DATA_DIR}/coco/train2014"}
SEG_MASK_FOLDERS=${SEG_MASK_FOLDERS:-"${DATA_DIR}/grefcoco_masks"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}
EXP_NAME=${EXP_NAME:-"Qwen2.5-VL-3B-Instruct-rec"}
TASK_TYPE=${TASK_TYPE:-"rec"}
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-12349}
# Weights & Biases logging overrides (optional)
WANDB_PROJECT=${WANDB_PROJECT:-"vlm-r1-rec"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-""}
WANDB_MODE=${WANDB_MODE:-""}  # set to "offline" or "disabled" if needed
WANDB_API_KEY=${WANDB_API_KEY:-"68f64368bb0a962b1d9227fe4b9da611e8f3c9c7"}        # optional; set to log in automatically
# ----------------------------------------------------------------------------

export REPO_HOME="$(cd "${REPO_HOME}" && pwd)"
export DEBUG_MODE=${DEBUG_MODE:-"false"}
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
mkdir -p "$(dirname "${LOG_PATH}")"

export WANDB_PROJECT
[[ -n "${WANDB_ENTITY}" ]] && export WANDB_ENTITY
[[ -n "${WANDB_RUN_GROUP}" ]] && export WANDB_RUN_GROUP
[[ -n "${WANDB_MODE}" ]] && export WANDB_MODE
[[ -n "${WANDB_API_KEY}" ]] && export WANDB_API_KEY

torchrun --nproc_per_node="${NUM_GPUS}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port="${MASTER_PORT}" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir "${REPO_HOME}/checkpoints/rl/${EXP_NAME}" \
    --resume_from_checkpoint True \
    --model_name_or_path "${MODEL_PATH}" \
    --data_file_paths "${DATA_PATHS}" \
    --image_folders "${IMAGE_FOLDERS}" \
    --seg_mask_folders "${SEG_MASK_FOLDERS}" \
    --is_reward_customized_from_vlm_module True \
    --task_type "${TASK_TYPE}" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
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
    --dataset-name grefcoco \
    --deepspeed "${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json"

echo "GRPO training launched for ${EXP_NAME}"
