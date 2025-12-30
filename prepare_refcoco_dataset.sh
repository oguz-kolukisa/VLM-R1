#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configurable paths -----------------------------------------------------
DATA_DIR=${DATA_DIR:-"${REPO_HOME}/data"}
REFER_REPO=${REFER_REPO:-"${DATA_DIR}/refer"}
REFER_DATA_ROOT=${REFER_DATA_ROOT:-"${REFER_REPO}/data"}
COCO_ROOT=${COCO_ROOT:-"${DATA_DIR}/coco"}
OUT_BASE=${OUT_BASE:-"${DATA_DIR}/refcoco_exports"}
DATASET=${DATASET:-"refcocog"}
SPLIT_BY=${SPLIT_BY:-"umd"}   # recommended split strategy for refcocog
SPLIT=${SPLIT:-"all"}         # comma list or "all" to cover train/val/test
# ----------------------------------------------------------------------------

mkdir -p "${DATA_DIR}" "${OUT_BASE}"

# Clone REFER repo if missing
if [[ ! -d "${REFER_REPO}/.git" ]]; then
    echo "Cloning REFER repo into ${REFER_REPO}"
    git clone https://github.com/lichengunc/refer.git "${REFER_REPO}"
else
    echo "Updating REFER repo in ${REFER_REPO}"
    git -C "${REFER_REPO}" pull --ff-only
fi

export DATA_DIR REFER_REPO REFER_DATA_ROOT COCO_ROOT OUT_BASE DATASET SPLIT_BY SPLIT

# Download COCO images + annotations (skip if already extracted)
python - <<'PY'
from pathlib import Path
import os

from torchvision.datasets.utils import download_url, extract_archive

coco_root = Path(os.environ["COCO_ROOT"])
coco_root.mkdir(parents=True, exist_ok=True)

archives = {
    "train2014.zip": {
        "url": "http://images.cocodataset.org/zips/train2014.zip",
        "check": coco_root / "train2014",
    },
    "val2014.zip": {
        "url": "http://images.cocodataset.org/zips/val2014.zip",
        "check": coco_root / "val2014",
    },
    "annotations_trainval2014.zip": {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        # pick a specific file that should exist after extraction
        "check": coco_root / "annotations" / "instances_train2014.json",
    },
}

for filename, meta in archives.items():
    check_path = meta["check"]
    zip_path = coco_root / filename

    if check_path.exists():
        print(f"[skip] {filename}: already present ({check_path})")
        continue

    if not zip_path.exists():
        print(f"[download] {filename}")
        download_url(meta["url"], root=str(coco_root), filename=filename)
    else:
        print(f"[have zip] {filename}: {zip_path}")

    print(f"[extract] {filename}")
    extract_archive(str(zip_path), to_path=str(coco_root))
PY


RUN_EXPORT=${RUN_EXPORT:-"1"}  # set to 0 to print commands only

OUT_SUFFIX="${SPLIT_BY:-auto}"
IFS=',' read -r -a REQUESTED_SPLITS <<< "${SPLIT}"
if [[ ${#REQUESTED_SPLITS[@]} -eq 1 && "${REQUESTED_SPLITS[0]}" == "all" ]]; then
    REQUESTED_SPLITS=(train val test)
fi

for CURRENT_SPLIT in "${REQUESTED_SPLITS[@]}"; do
    CURRENT_SPLIT=$(echo "${CURRENT_SPLIT}" | xargs)
    [[ -z "${CURRENT_SPLIT}" ]] && continue
    OUT_DIR="${OUT_BASE}/${DATASET}_${OUT_SUFFIX}_${CURRENT_SPLIT}"
    EXPORT_ARGS=(
        --refer_repo "${REFER_REPO}"
        --refer_data_root "${DATA_DIR}"
        --dataset "${DATASET}"
        --split "${CURRENT_SPLIT}"
        --out_dir "${OUT_DIR}"
    )
    if [[ -n "${SPLIT_BY}" ]]; then
        EXPORT_ARGS+=(--splitBy "${SPLIT_BY}")
    fi
    if [[ "${RUN_EXPORT}" == "0" ]]; then
        echo "[dry-run] ${DATASET}/${SPLIT_BY:-auto}/${CURRENT_SPLIT} -> ${OUT_DIR}"
        printf '  python "%s/scripts/export_refcoco_masks.py" %s\n' \
            "${REPO_HOME}" "${EXPORT_ARGS[*]}"
    else
        echo "Exporting ${DATASET}/${SPLIT_BY:-auto}/${CURRENT_SPLIT} -> ${OUT_DIR}"
        python "${REPO_HOME}/scripts/export_refcoco_masks.py" "${EXPORT_ARGS[@]}"
    fi
done

if [[ "${RUN_EXPORT}" == "0" ]]; then
    echo "RefCOCO export commands printed only (RUN_EXPORT=0)."
else
    echo "RefCOCO exports complete under ${OUT_BASE}"
fi
