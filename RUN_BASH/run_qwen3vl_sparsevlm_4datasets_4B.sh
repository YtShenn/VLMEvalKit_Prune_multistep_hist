#!/usr/bin/env bash
set -euo pipefail

# Shared SparseVLM runner for the four dataset-specific entry points below.
# Override DATASETS to run a custom subset.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
MODEL="${MODEL:-Qwen3-VL-4B-Instruct-SparseVLM}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
WORK_DIR="${WORK_DIR:-OUTPUT_SPARSEVLM/4B_keep${SPARSEVLM_RETAIN_RATIO:-0.7}}"
DATASETS="${DATASETS:-AITW_all Mind2Web_test_task GUIOdyssey_high_task_split AndroidControl_Curated_High_Task_Improved}"

export SPARSEVLM_ENABLED="${SPARSEVLM_ENABLED:-1}"
export SPARSEVLM_LAYERS="${SPARSEVLM_LAYERS:-3,6,15}"
export SPARSEVLM_RETAIN_RATIO="${SPARSEVLM_RETAIN_RATIO:-0.7}"
export SPARSEVLM_VERSION="${SPARSEVLM_VERSION:-1_0}"
export SPARSEVLM_SCOPE="${SPARSEVLM_SCOPE:-all}" # all/history/current
export SPARSEVLM_MAX_HISTORY_FRAMES="${SPARSEVLM_MAX_HISTORY_FRAMES:-4}"
export SPARSEVLM_DEBUG="${SPARSEVLM_DEBUG:-0}"

# Unified <=4-history protocol used by the existing four dataset adapters.
export GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS="${GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS:-1}"
export GUI_ODYSSEY_MAX_HISTORY_IMAGES="${GUI_ODYSSEY_MAX_HISTORY_IMAGES:-4}"
export ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS="${ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS:-1}"
export ANDROID_CONTROL_MAX_HISTORY_IMAGES="${ANDROID_CONTROL_MAX_HISTORY_IMAGES:-4}"
export ANDROID_CONTROL_SEQUENTIAL_ORDER="${ANDROID_CONTROL_SEQUENTIAL_ORDER:-1}"
export ANDROID_CONTROL_HISTORY_KEEP_SYSTEM_PROMPT="${ANDROID_CONTROL_HISTORY_KEEP_SYSTEM_PROMPT:-1}"
export ANDROID_CONTROL_OUTPUT_PROTOCOL="${ANDROID_CONTROL_OUTPUT_PROTOCOL:-auto}"
export AITW_HIS_NUM="${AITW_HIS_NUM:-4}"
export MIND2WEB_HIS_NUM="${MIND2WEB_HIS_NUM:-4}"
# Keep this a pure SparseVLM baseline.
export QWEN3VL_ENABLE_ATTN_PRUNE=0
export QWEN3VL_ENABLE_ROI_PRUNE=0
export QWEN3VL_ENABLE_TEMPLATE_PREFILL=0
export QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE="${QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE:-1}"
export AITW_STATE_PACKET_ENABLE=0
export MIND2WEB_STATE_PACKET_ENABLE=0
# AndroidControl ground truth and its official evaluator use coordinates in
# the original current-screenshot pixel space. Qwen3-VL commonly emits the
# prompt's 0--1000 coordinates, so convert only AndroidControl responses
# before evaluation. This has no effect on the other three datasets.
export QWEN3VL_ANDROID_DENORM_ON_INFER="${QWEN3VL_ANDROID_DENORM_ON_INFER:-1}"
export QWEN3VL_ANDROID_DENORM_BASE="${QWEN3VL_ANDROID_DENORM_BASE:-1000}"
export QWEN3VL_ATTN_PRUNE_USE_CACHE=1
export VLM_TIMING="${VLM_TIMING:-1}"
export QWEN3VL_RUNTIME_TRACKING="${QWEN3VL_RUNTIME_TRACKING:-1}"
export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-off}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-5}"
export VLM_EVAL_SAMPLE_COUNT="${VLM_EVAL_SAMPLE_COUNT:-20}"

export AITW_ANN_ROOT="${AITW_ANN_ROOT:-/public/ytshen/aitw_data/aitw_annots}"
export AITW_IMAGE_ROOT="${AITW_IMAGE_ROOT:-/public/ytshen/aitw_data/aitw_images}"
export AITW_SPLIT="${AITW_SPLIT:-test}"
export MIND2WEB_DATA_ROOT="${MIND2WEB_DATA_ROOT:-/public/ytshen/Mind2Web}"
export MIND2WEB_ANN_ROOT="${MIND2WEB_ANN_ROOT:-${MIND2WEB_DATA_ROOT}/mind2web_annots}"
if [[ -z "${MIND2WEB_IMAGE_ROOT:-}" && -d "${MIND2WEB_DATA_ROOT}/mind2web_images/ming2web_images" ]]; then
  export MIND2WEB_IMAGE_ROOT="${MIND2WEB_DATA_ROOT}/mind2web_images/ming2web_images"
else
  export MIND2WEB_IMAGE_ROOT="${MIND2WEB_IMAGE_ROOT:-${MIND2WEB_DATA_ROOT}/mind2web_images}"
fi
export DATA_ROOT="${DATA_ROOT:-/public/ljkang/datasets/GUIOdyssey}"
export ANDROID_CONTROL_CURATED_EVAL_MODE="${ANDROID_CONTROL_CURATED_EVAL_MODE:-official}"
export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/public/ytshen/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/public/ytshen/AndroidControl_Curated/images}"

mkdir -p "${WORK_DIR}"; stamp="$(date +%Y%m%d_%H%M%S)"
for dataset in ${DATASETS}; do
  out="${WORK_DIR}/${dataset}"; mkdir -p "${out}"; log="${WORK_DIR}/run_${stamp}_${dataset}.log"
  echo "[SparseVLM] dataset=${dataset} layers=${SPARSEVLM_LAYERS} retain=${SPARSEVLM_RETAIN_RATIO} scope=${SPARSEVLM_SCOPE}" | tee -a "${log}"
  if [[ "${NPROC_PER_NODE}" == 1 ]]; then
    "${PYTHON_BIN}" run.py --data "${dataset}" --model "${MODEL}" --work-dir "${out}" --mode all 2>&1 | tee -a "${log}"
  else
    "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" run.py --data "${dataset}" --model "${MODEL}" --work-dir "${out}" --mode all 2>&1 | tee -a "${log}"
  fi
done
