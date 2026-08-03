#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1,3,5,7
NPROC_PER_NODE=4

MODEL=${MODEL:-UI-TARS-1.5-7B}
WORK_DIR=${WORK_DIR:-OUTPUT/outputs_ui_tars15_gui_android}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# UI-TARS local checkpoint
export EVAL_MODEL=${EVAL_MODEL:-/mnt/storage/users/ytshen_data/UI-TARS-1.5-7B}

# GUIOdyssey coord normalization switch: 1=normalize to [0,1000], 0=keep raw coords
export UITARS_GUIODYSSEY_COORD_NORM=1

# AndroidControl paths
export ANDROID_CONTROL_CURATED_ROOT=${ANDROID_CONTROL_CURATED_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated}
export ANDROID_CONTROL_CURATED_IMAGE_ROOT=${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images}

# GUIOdyssey root: contains test_anno/ and screenshots/
export GUI_ODYSSEY_ROOT=${GUI_ODYSSEY_ROOT:-/mnt/storage2/users/ytshen_data/GUIOdyssey}

mkdir -p "${WORK_DIR}"

DS_LIST=(
#   "AndroidControl_Curated_High_Task_Improved"
  "GUIOdyssey_high_task_split"
)

for ds in "${DS_LIST[@]}"; do
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port=29602 run.py \
    --data "${ds}" \
    --model "${MODEL}" \
    --work-dir "${WORK_DIR}_${ds}" \
    --mode all \
    --reuse \
    2>&1 | tee -a "run_output_${TIMESTAMP}_ui_tars15_${ds}.log"
done
