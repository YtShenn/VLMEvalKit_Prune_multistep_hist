#!/usr/bin/env bash
set -euo pipefail

# Visualize sampled multi-step tasks for GUIOdyssey and AndroidControl.
# Each sampled task is exported as one folder containing per-step annotated images.
#
# Example:
#   bash RUN_BASH/run_visualize_multistep_gui_tasks.sh
#
# Override defaults:
#   DATASETS="GUIOdyssey_high_task_split AndroidControl_Curated_High_Task_Improved" \
#   NUM_TASKS=10 \
#   SEED=0 \
#   OUTPUT_DIR=OUTPUT/task_visualizations \
#   bash RUN_BASH/run_visualize_multistep_gui_tasks.sh
export CUDA_VISIBLE_DEVICES=1
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="utils/visualize_multistep_gui_tasks.py"

export GUI_ODYSSEY_ROOT="${GUI_ODYSSEY_ROOT:-/mnt/storage2/users/ytshen_data/GUIOdyssey}"
export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images}"

DATASETS="${DATASETS:-GUIOdyssey_high_task_split AndroidControl_Curated_High_Task_Improved}"
NUM_TASKS="${NUM_TASKS:-10}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-OUTPUT/task_visualizations}"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[ERROR] Missing script: ${SCRIPT_PATH}"
  exit 1
fi

if [[ ! -d "${GUI_ODYSSEY_ROOT}/test_anno" ]]; then
  echo "[ERROR] Missing GUIOdyssey annotations: ${GUI_ODYSSEY_ROOT}/test_anno"
  exit 1
fi

if [[ ! -d "${ANDROID_CONTROL_CURATED_ROOT}/benchmark_resource" ]]; then
  echo "[ERROR] Missing AndroidControl benchmark_resource: ${ANDROID_CONTROL_CURATED_ROOT}/benchmark_resource"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[Run] SCRIPT_PATH=${SCRIPT_PATH}"
echo "[Run] DATASETS=${DATASETS}"
echo "[Run] NUM_TASKS=${NUM_TASKS}"
echo "[Run] SEED=${SEED}"
echo "[Run] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[Run] GUI_ODYSSEY_ROOT=${GUI_ODYSSEY_ROOT}"
echo "[Run] ANDROID_CONTROL_CURATED_ROOT=${ANDROID_CONTROL_CURATED_ROOT}"
echo "[Run] ANDROID_CONTROL_CURATED_IMAGE_ROOT=${ANDROID_CONTROL_CURATED_IMAGE_ROOT}"

"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --datasets ${DATASETS} \
  --num_tasks "${NUM_TASKS}" \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_DIR}"
