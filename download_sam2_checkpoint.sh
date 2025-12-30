#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_HOME}"

# --- User editable values ---------------------------------------------------
SAM2_DIR=${SAM2_DIR:-"${REPO_HOME}/checkpoints/sam2"}
SAM2_MODEL_NAME=${SAM2_MODEL_NAME:-"sam2_hiera_large"}
# ----------------------------------------------------------------------------

mkdir -p "${SAM2_DIR}"

case "${SAM2_MODEL_NAME}" in
    sam2_hiera_tiny)
        SAM2_URL="https://dl.fbaipublicfiles.com/SAM2/models/sam2_hiera_tiny.pt"
        ;;
    sam2_hiera_small)
        SAM2_URL="https://dl.fbaipublicfiles.com/SAM2/models/sam2_hiera_small.pt"
        ;;
    sam2_hiera_base)
        SAM2_URL="https://dl.fbaipublicfiles.com/SAM2/models/sam2_hiera_base.pt"
        ;;
    sam2_hiera_base_plus)
        SAM2_URL="https://dl.fbaipublicfiles.com/SAM2/models/sam2_hiera_base_plus.pt"
        ;;
    sam2_hiera_large)
        SAM2_URL="https://dl.fbaipublicfiles.com/SAM2/models/sam2_hiera_large.pt"
        ;;
    *)
        echo "Unsupported SAM2 model '${SAM2_MODEL_NAME}'."
        echo "Supported models: sam2_hiera_tiny, sam2_hiera_small, sam2_hiera_base, sam2_hiera_base_plus, sam2_hiera_large."
        exit 1
        ;;
esac

DEST_PATH="${SAM2_DIR}/${SAM2_MODEL_NAME}.pt"

if [[ -f "${DEST_PATH}" ]]; then
    echo "Checkpoint already exists at ${DEST_PATH}"
else
    echo "Downloading ${SAM2_MODEL_NAME} from ${SAM2_URL}"
    curl -L -o "${DEST_PATH}" "${SAM2_URL}"
fi

echo "SAM2 checkpoint saved to ${DEST_PATH}"
echo "Set SAM2_CKPT=${DEST_PATH} before running training or evaluation."
