#!/usr/bin/env bash
# Compare the original teacher-forced history ROI with a free-running ROI
# centred on the model's actual prior prediction.  Both runs use one process:
# free-running history has a causal per-trajectory dependency.
set -euo pipefail

RUNNER="${RUNNER:-RUN_BASH/run_android_control_history_state_packet_qwen3_5.sh}"
DATASET="${DATASET:-AndroidControl_Curated_High_Task_Improved}"
TAG_BASE="${TAG_BASE:-hist_roi_gt_vs_prediction}"
COMPARISON_ROOT="${COMPARISON_ROOT:-OUTPUT_ROI_SOURCE/${TAG_BASE}}"
PYTHON_BIN="${PYTHON_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/python}"

if [[ ! -f "${RUNNER}" ]]; then
  echo "[ERROR] runner not found: ${RUNNER}" >&2
  exit 1
fi

for source in gt prediction; do
  echo "[ROI comparison] source=${source}"
  ANDROID_CONTROL_HISTORY_ACTION_SOURCE="${source}" \
  TAG="${TAG_BASE}_${source}" \
  WORK_DIR="${COMPARISON_ROOT}/${source}" \
  DATASET_LIST_OVERRIDE="${DATASET}" \
  bash "${RUNNER}"
done

# The base runner currently declares its dataset list internally.  Therefore
# this comparison helper is intended for its default AndroidControl dataset.
# Its summaries are written directly below these two paths.
GT_SUMMARY="${COMPARISON_ROOT}/gt/${DATASET}/summary.json"
PRED_SUMMARY="${COMPARISON_ROOT}/prediction/${DATASET}/summary.json"
if [[ -f "${GT_SUMMARY}" && -f "${PRED_SUMMARY}" ]]; then
  "${PYTHON_BIN}" utils/compare_summary_json.py \
    "${GT_SUMMARY}" "${PRED_SUMMARY}" \
    --labels gt_roi predicted_roi \
    --fields android_history_action_source avg_total_wall_s avg_infer_wall_s avg_decode_tokens avg_decode_steps state_packet_total_packet_estimated_tokens state_packet_total_original_estimated_tokens \
    --output "${COMPARISON_ROOT}/gt_vs_prediction.md"
  echo "[ROI comparison] report: ${COMPARISON_ROOT}/gt_vs_prediction.md"
else
  echo "[ROI comparison] summaries not found; compare manually once runs finish:" >&2
  echo "  ${GT_SUMMARY}" >&2
  echo "  ${PRED_SUMMARY}" >&2
fi
