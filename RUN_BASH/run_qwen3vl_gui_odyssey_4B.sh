#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash RUN_BASH/run_qwen3vl_gui_odyssey.sh
# Optional env vars:
#   GUI_ODYSSEY_ROOT=/home/ytshen/storage_net2/GUI-Odyssey-master
#   MODEL=Qwen3-VL-8B-Instruct
#   DATA=GUIOdyssey_high_random_split
#   NPROC_PER_NODE=1
#   WORK_DIR=OUTPUT/outputs_qwen3vl_guiodyssey
#   PREPARE_DATA=1

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=6
#几张卡跑
NPROC_PER_NODE=1

# 随机数种子
export SEED=42

DATA_ROOT="${DATA_ROOT:-/mnt/storage2/users/ytshen_data/GUIOdyssey}"
export DATA_ROOT
MODEL="Qwen3-VL-4B-Instruct"
DS_LIST=("low_task_split") #"high_task_split" 

WORK_DIR="OUTPUT/outputs_qwen3vl_guiodyssey_4B"
PREPARE_DATA=0

#计时
export VLM_TIMING=1
export VLM_TIMING_VERBOSE=1
export VLM_TIMING_SYNC=1
export VLM_STAGE_TIMING=1
export VLM_STAGE_TIMING_DEVICE=auto
export VLM_STAGE_TIMING_SYNC=0  

export VLM_PROGRESS_INTERVAL=0
export VLM_PROGRESS_ACC=0

# ds=high_task_split

if [[ "${PREPARE_DATA}" == "1" ]]; then
  echo "[GUIOdyssey] Preparing test_anno json files from raw annotations..."
  echo "[GUIOdyssey] Data root: ${DATA_ROOT}"
  python vlmeval/dataset/GUI_Odyssey/format_converter.py \
    --data-root "${DATA_ROOT}" --his_len 4 --level high --type standard
  python vlmeval/dataset/GUI_Odyssey/format_converter.py \
    --data-root "${DATA_ROOT}" --his_len 4 --level low --type standard
fi

if [[ ! -d "${DATA_ROOT}/screenshots" ]]; then
  echo "[ERROR] Missing screenshots directory: ${DATA_ROOT}/screenshots"
  exit 1
fi

PNG_COUNT=$(find "${DATA_ROOT}/screenshots" -maxdepth 1 -type f -name '*.png' | wc -l)
if [[ "${PNG_COUNT}" -eq 0 ]]; then
  echo "[ERROR] No screenshot png files found in ${DATA_ROOT}/screenshots"
  echo "Please download GUI-Odyssey screenshots first, then rerun."
  exit 1
fi

# mkdir -p "${WORK_DIR}"
# if [[ ! -f "${DATA_ROOT}/test_anno/${ds}.json" ]]; then
#     echo "[ERROR] Missing ${DATA_ROOT}/test_anno/${ds}.json"
#     echo "Please run GUIOdyssey data preprocessing first."
#     exit 1
#   fi

#   DATA="GUIOdyssey_${ds}"
#   mkdir -p "${WORK_DIR}_${ds}"

#   TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
#   echo "[Run] MODEL=${MODEL} DATA=${DATA} NPROC_PER_NODE=${NPROC_PER_NODE} WORK_DIR=${WORK_DIR}_${ds}"
#   # torchrun -m debugpy --listen localhost:58735 --wait-for-client \
#   # python -u -m debugpy --listen localhost:5388 --wait-for-client run.py \
#   torchrun --nproc_per_node="${NPROC_PER_NODE}" run.py \
#     --data "${DATA}" \
#     --model "${MODEL}" \
#     --work-dir "${WORK_DIR}_${ds}" \
#     --mode all \
#     2>&1 | tee -a "run_output_${TIMESTAMP}_gui_odyssey_${ds}_4B.log"

for ds in "${DS_LIST[@]}"; do
  if [[ ! -f "${DATA_ROOT}/test_anno/${ds}.json" ]]; then
    echo "[ERROR] Missing ${DATA_ROOT}/test_anno/${ds}.json"
    echo "Please run GUIOdyssey data preprocessing first."
    exit 1
  fi

  DATA="GUIOdyssey_${ds}"
  mkdir -p "${WORK_DIR}_${ds}"

  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  echo "[Run] MODEL=${MODEL} DATA=${DATA} NPROC_PER_NODE=${NPROC_PER_NODE} WORK_DIR=${WORK_DIR}_${ds}"
  # torchrun -m debugpy --listen localhost:58735 --wait-for-client \
  # python -u -m debugpy --listen localhost:5388 --wait-for-client run.py \
  # python -u run.py \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" run.py \
    --data "${DATA}" \
    --model "${MODEL}" \
    --work-dir "${WORK_DIR}_${ds}" \
    --mode all \
    2>&1 | tee -a "run_output_${TIMESTAMP}_gui_odyssey_${ds}_4B.log"
done
