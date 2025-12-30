#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_HOME}/src/open-r1-multimodal"

# --- User editable paths ----------------------------------------------------
DATA_DIR=${DATA_DIR:-"${REPO_HOME}/data"}
REFCOCO_EXPORT_BASE=${REFCOCO_EXPORT_BASE:-"${DATA_DIR}/refcoco_exports"}
COCO_IMAGE_ROOT=${COCO_IMAGE_ROOT:-"${DATA_DIR}/coco"}
DATA_PATHS=${DATA_PATHS:-""}
IMAGE_FOLDERS=${IMAGE_FOLDERS:-""}
SEG_MASK_FOLDERS=${SEG_MASK_FOLDERS:-""}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}
EXP_NAME=${EXP_NAME:-"Qwen2.5-VL-3B-Instruct-rec"}
TASK_TYPE=${TASK_TYPE:-"segmednt"}
NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-12349}
# Weights & Biases logging overrides (optional)
WANDB_PROJECT=${WANDB_PROJECT:-"vlm-r1-rec"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-""}
WANDB_MODE=${WANDB_MODE:-""}  # set to "offline" or "disabled" if needed
WANDB_API_KEY=${WANDB_API_KEY:-""}        # optional; set to log in automatically
# ----------------------------------------------------------------------------

export REPO_HOME="$(cd "${REPO_HOME}" && pwd)"
export PYTHONPATH="${REPO_HOME}/src:${REPO_HOME}/src/open-r1-multimodal/src:${PYTHONPATH:-}"
export DEBUG_MODE=${DEBUG_MODE:-"false"}
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
mkdir -p "$(dirname "${LOG_PATH}")"
mkdir -p "${DATA_DIR}"

if [[ -z "${DATA_PATHS}" ]]; then
    if [[ ! -d "${REFCOCO_EXPORT_BASE}" ]]; then
        echo "No refcoco exports found at ${REFCOCO_EXPORT_BASE}. Run prepare_refcoco_dataset.sh first."
        exit 1
    fi
    mapfile -t AUTO_DATA_PATHS < <(find "${REFCOCO_EXPORT_BASE}" -maxdepth 2 -name "metadata.jsonl" | sort)
    if [[ ${#AUTO_DATA_PATHS[@]} -eq 0 ]]; then
        echo "No metadata.jsonl files detected under ${REFCOCO_EXPORT_BASE}."
        exit 1
    fi
    DATA_PATHS=$(IFS=:; echo "${AUTO_DATA_PATHS[*]}")
    if [[ -z "${IMAGE_FOLDERS}" ]]; then
        IMAGE_ARRAY=()
        for _ in "${AUTO_DATA_PATHS[@]}"; do
            IMAGE_ARRAY+=("${COCO_IMAGE_ROOT}")
        done
        IMAGE_FOLDERS=$(IFS=:; echo "${IMAGE_ARRAY[*]}")
    fi
fi

if [[ -z "${IMAGE_FOLDERS}" ]]; then
    echo "IMAGE_FOLDERS not set and no auto-discovery available."
    exit 1
fi

if [[ -n "${SEG_MASK_FOLDERS}" ]]; then
    IFS=':' read -r -a SEG_MASK_ARRAY <<< "${SEG_MASK_FOLDERS}"
    IFS=':' read -r -a DATA_PATH_ARRAY <<< "${DATA_PATHS}"
    if [[ ${#SEG_MASK_ARRAY[@]} -ne 0 && ${#SEG_MASK_ARRAY[@]} -ne ${#DATA_PATH_ARRAY[@]} ]]; then
        echo "SEG_MASK_FOLDERS count (${#SEG_MASK_ARRAY[@]}) must match DATA_PATHS count (${#DATA_PATH_ARRAY[@]})."
        exit 1
    fi
fi

export WANDB_PROJECT
[[ -n "${WANDB_ENTITY}" ]] && export WANDB_ENTITY
[[ -n "${WANDB_RUN_GROUP}" ]] && export WANDB_RUN_GROUP
[[ -n "${WANDB_MODE}" ]] && export WANDB_MODE
[[ -n "${WANDB_API_KEY}" ]] && export WANDB_API_KEY

EXTRA_SEG_MASK_ARGS=()
if [[ -n "${SEG_MASK_FOLDERS}" ]]; then
    EXTRA_SEG_MASK_ARGS+=(--seg_mask_folders "${SEG_MASK_FOLDERS}")
fi

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
    "${EXTRA_SEG_MASK_ARGS[@]}" \
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
    --reward_funcs mask_iou format \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name grefcoco \
    --deepspeed "${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json"

echo "GRPO training launched for ${EXP_NAME}"
