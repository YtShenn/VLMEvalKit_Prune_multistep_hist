#!/usr/bin/env bash
# Compare teacher-forced and free-running history ROIs for AITW.
set -euo pipefail

RUNNER="${RUNNER:-RUN_BASH/run_qwen3vl_aitw_4B_state_packet_fast_decode_only.sh}"
DATASET="${DATASET:-AITW_all}"
TAG_BASE="${TAG_BASE:-hist_roi_gt_vs_prediction}"
COMPARISON_ROOT="${COMPARISON_ROOT:-OUTPUT_ROI_SOURCE/aitw/${TAG_BASE}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

[[ -f "${RUNNER}" ]] || { echo "[ERROR] runner not found: ${RUNNER}" >&2; exit 1; }

for source in prediction; do
  echo "[ROI comparison] dataset=${DATASET} source=${source}"
  AITW_HISTORY_ACTION_SOURCE="${source}" \
  QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE=1 \
  TAG="${TAG_BASE}_${source}" \
  WORK_DIR="${COMPARISON_ROOT}/${source}" \
  DATASET_LIST_OVERRIDE="${DATASET}" \
  bash "${RUNNER}"
done

GT_SUMMARY="${COMPARISON_ROOT}/gt/${DATASET}/summary.json"
PRED_SUMMARY="${COMPARISON_ROOT}/prediction/${DATASET}/summary.json"
if [[ -f "${GT_SUMMARY}" && -f "${PRED_SUMMARY}" ]]; then
  "${PYTHON_BIN}" utils/compare_summary_json.py "${GT_SUMMARY}" "${PRED_SUMMARY}" \
    --labels gt_roi predicted_roi \
    --fields aitw_history_action_source avg_total_wall_s avg_infer_wall_s avg_decode_tokens avg_decode_steps state_packet_total_packet_estimated_tokens state_packet_total_original_estimated_tokens \
    --output "${COMPARISON_ROOT}/gt_vs_prediction.md"
  echo "[ROI comparison] report: ${COMPARISON_ROOT}/gt_vs_prediction.md"
fi
