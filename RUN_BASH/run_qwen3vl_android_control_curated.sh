#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash RUN_BASH/run_qwen3vl_android_control_curated.sh
# Optional env vars:
#   MODEL=Qwen3-VL-8B-Instruct
#   WORK_DIR=OUTPUT/outputs_qwen3vl_android_control_curated
#   CUDA_VISIBLE_DEVICES=0
#   ANDROID_CONTROL_CURATED_ROOT=/home/ytshen/storage_net2/AndroidControl_Curated-main/src
#   ANDROID_CONTROL_CURATED_IMAGE_ROOT=/path/to/android_control_images_root

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0

#计时
export VLM_TIMING=1
export VLM_TIMING_VERBOSE=1
export VLM_TIMING_SYNC=1
export VLM_STAGE_TIMING=1
export VLM_STAGE_TIMING_DEVICE=auto
export VLM_STAGE_TIMING_SYNC=0  

# 随机数种子
export SEED=42

MODEL="Qwen3-VL-8B-Instruct"
WORK_DIR="OUTPUT/outputs_qwen3vl_android_control_curated_8B"

export ANDROID_CONTROL_CURATED_ROOT="/mnt/storage2/users/ytshen_data/AndroidControl_Curated"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images"

if [[ ! -d "${ANDROID_CONTROL_CURATED_ROOT}/benchmark_resource" ]]; then
  echo "[ERROR] Missing benchmark_resource under ANDROID_CONTROL_CURATED_ROOT=${ANDROID_CONTROL_CURATED_ROOT}"
  exit 1
fi

mkdir -p "${WORK_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "[Run] MODEL=${MODEL} WORK_DIR=${WORK_DIR}"
echo "[Run] ANDROID_CONTROL_CURATED_ROOT=${ANDROID_CONTROL_CURATED_ROOT}"
echo "[Run] ANDROID_CONTROL_CURATED_IMAGE_ROOT=${ANDROID_CONTROL_CURATED_IMAGE_ROOT}"

python -u run.py \
  --data \
    AndroidControl_Curated_Low_Point \
    AndroidControl_Curated_High_Point \
    AndroidControl_Curated_Low_BBox \
    AndroidControl_Curated_High_BBox \
    AndroidControl_Curated_High_Task_Improved \
  --model "${MODEL}" \
  --work-dir "${WORK_DIR}" \
  --mode all \
  2>&1 | tee -a "run_output_${TIMESTAMP}_android_control_curated.log"
