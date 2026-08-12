#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash RUN_BASH/run_android_control_official_eval_from_xlsx.sh \
#     OUTPUT/foo/AndroidControl_Curated_High_Point/model.xlsx \
#     OUTPUT/foo/AndroidControl_Curated_High_Task_Improved/model.xlsx

if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash $0 <android_control_xlsx> [more_xlsx ...]"
  exit 1
fi

python utils/eval_android_control_official.py --xlsx "$@"
