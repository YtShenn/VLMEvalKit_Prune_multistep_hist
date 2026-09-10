#!/usr/bin/env bash
set -euo pipefail

# Faithful DivPrune: all four old->new history screenshots AND the current
# screenshot form one visual candidate pool before Qwen3-VL decoder layer 0.
# Example: DATASETS=AndroidControl_Curated_High_Task_Improved NPROC_PER_NODE=1 \
#   bash RUN_BASH/run_qwen3vl_divprune_4B.sh
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SEED="${SEED:-42}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MODEL="${MODEL:-Qwen3-VL-4B-Instruct-DivPrune}"
DATASETS="${DATASETS:-AndroidControl_Curated_High_Task_Improved}"

export DIVPRUNE_ENABLED="${DIVPRUNE_ENABLED:-1}"
export DIVPRUNE_KEEP_RATIO="${DIVPRUNE_KEEP_RATIO:-0.098}"
export DIVPRUNE_SCOPE="${DIVPRUNE_SCOPE:-global_all_visual}"
export DIVPRUNE_DEBUG="${DIVPRUNE_DEBUG:-0}"
WORK_DIR="${WORK_DIR:-OUTPUT_DIVPRUNE/4B_hist4_keep${DIVPRUNE_KEEP_RATIO}}"

# Preserve the complete four-step history context. DivPrune only reduces the
# visual token representation after the vision projector; no screenshots or
# history text are removed.
export GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS="${GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS:-1}"
export GUI_ODYSSEY_MAX_HISTORY_IMAGES="${GUI_ODYSSEY_MAX_HISTORY_IMAGES:-4}"
export ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS="${ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS:-1}"
export ANDROID_CONTROL_MAX_HISTORY_IMAGES="${ANDROID_CONTROL_MAX_HISTORY_IMAGES:-4}"
export AITW_HIS_NUM="${AITW_HIS_NUM:-4}"
export MIND2WEB_HIS_NUM="${MIND2WEB_HIS_NUM:-4}"

# Keep this baseline isolated from unrelated pruning/decoding accelerators.
export QWEN3VL_ENABLE_ATTN_PRUNE=0
export QWEN3VL_ENABLE_ROI_PRUNE=0
export QWEN3VL_ENABLE_TEMPLATE_PREFILL=0
export QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE=0
# AndroidControl prompts use 0--1000 coordinates, while its official
# evaluator uses absolute pixels on the original current screenshot. The
# model's dataset-gated postprocessor performs this conversion only for
# AndroidControl, so these flags do not alter AITW/Mind2Web/GUIOdyssey.
export QWEN3VL_ANDROID_DENORM_ON_INFER="${QWEN3VL_ANDROID_DENORM_ON_INFER:-1}"
export QWEN3VL_ANDROID_DENORM_BASE="${QWEN3VL_ANDROID_DENORM_BASE:-1000}"
# Keep every comparison backend on the ordinary four-history-image input
# protocol. Otherwise a caller's AndroidControl state-packet setting can leak
# into AITW or Mind2Web and make the baseline inputs incomparable.
export AITW_STATE_PACKET_ENABLE=0
export MIND2WEB_STATE_PACKET_ENABLE=0
export QWEN3VL_RUNTIME_TRACKING="${QWEN3VL_RUNTIME_TRACKING:-1}"
export QWEN3VL_PROFILE_FLOPS="${QWEN3VL_PROFILE_FLOPS:-1}"
export VLM_TIMING="${VLM_TIMING:-1}"
export VLM_TIMING_VERBOSE="${VLM_TIMING_VERBOSE:-1}"

# Task-level small-sample mode. Keep it off for a full benchmark; a compact
# smoke run can use e.g. VLM_EVAL_SAMPLE_MODE=task VLM_EVAL_SAMPLE_TASKS=5
# VLM_EVAL_SAMPLE_COUNT=20. The evaluator owns the exact sampling semantics.
export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-task}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-5}"
export VLM_EVAL_SAMPLE_COUNT="${VLM_EVAL_SAMPLE_COUNT:-20}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"

# Dataset locations. Each value may be overridden by the launching shell.
export AITW_ANN_ROOT="${AITW_ANN_ROOT:-/mnt/storage2/Datasets/aitw_data/aitw_annots}"
export AITW_IMAGE_ROOT="${AITW_IMAGE_ROOT:-/mnt/storage2/Datasets/aitw_data/aitw_images}"
export AITW_SPLIT="${AITW_SPLIT:-test}"
export AITW_WITH_NO_HISTORY="${AITW_WITH_NO_HISTORY:-0}"
export AITW_SEMANTIC_ACTION_PROMPT="${AITW_SEMANTIC_ACTION_PROMPT:-0}"
export MIND2WEB_DATA_ROOT="${MIND2WEB_DATA_ROOT:-/mnt/storage2/Datasets/Mind2Web}"
export MIND2WEB_ANN_ROOT="${MIND2WEB_ANN_ROOT:-${MIND2WEB_DATA_ROOT}/mind2web_annots}"
if [[ -z "${MIND2WEB_IMAGE_ROOT:-}" && -d "${MIND2WEB_DATA_ROOT}/mind2web_images/ming2web_images" ]]; then
  export MIND2WEB_IMAGE_ROOT="${MIND2WEB_DATA_ROOT}/mind2web_images/ming2web_images"
else
  export MIND2WEB_IMAGE_ROOT="${MIND2WEB_IMAGE_ROOT:-${MIND2WEB_DATA_ROOT}/mind2web_images}"
fi
export MIND2WEB_WITH_NO_HISTORY="${MIND2WEB_WITH_NO_HISTORY:-0}"
export MIND2WEB_STRICT_OUTPUT_PROMPT="${MIND2WEB_STRICT_OUTPUT_PROMPT:-1}"
export DATA_ROOT="${DATA_ROOT:-/mnt/storage2/users/ytshen_data/GUIOdyssey}"
export ANDROID_CONTROL_CURATED_EVAL_MODE="${ANDROID_CONTROL_CURATED_EVAL_MODE:-official}"
export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images}"

require_dataset_files() {
  local dataset="$1"
  case "${dataset}" in
    AITW_*) [[ -f "${AITW_ANN_ROOT}/aitw_data_${AITW_SPLIT}.json" && -d "${AITW_IMAGE_ROOT}" ]] ;;
    Mind2Web_test_*) local split="${dataset#Mind2Web_test_}"; [[ -f "${MIND2WEB_ANN_ROOT}/mind2web_data_test_${split}.json" && -d "${MIND2WEB_IMAGE_ROOT}" ]] ;;
    GUIOdyssey_*) local split="${dataset#GUIOdyssey_}"; [[ -d "${DATA_ROOT}/screenshots" && -f "${DATA_ROOT}/test_anno/${split}.json" ]] ;;
    AndroidControl_Curated_*) [[ -d "${ANDROID_CONTROL_CURATED_ROOT}/benchmark_resource" && -d "${ANDROID_CONTROL_CURATED_IMAGE_ROOT}" ]] ;;
    *) echo "[ERROR] Unsupported dataset: ${dataset}"; return 1 ;;
  esac || { echo "[ERROR] Missing dataset files for ${dataset}"; return 1; }
}

mkdir -p "${WORK_DIR}"
for dataset in ${DATASETS}; do
  require_dataset_files "${dataset}"
  output_dir="${WORK_DIR}/${dataset}"
  mkdir -p "${output_dir}"
  echo "[DivPrune] dataset=${dataset} keep_ratio=${DIVPRUNE_KEEP_RATIO} scope=${DIVPRUNE_SCOPE} history_frames=4"
  echo "[Sample] mode=${VLM_EVAL_SAMPLE_MODE} tasks=${VLM_EVAL_SAMPLE_TASKS} count=${VLM_EVAL_SAMPLE_COUNT} seed=${VLM_EVAL_SAMPLE_SEED}"
  if [[ "${NPROC_PER_NODE}" == "1" ]]; then
    "${PYTHON_BIN}" run.py --data "${dataset}" --model "${MODEL}" --work-dir "${output_dir}" --mode all
  else
    "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" run.py --data "${dataset}" --model "${MODEL}" --work-dir "${output_dir}" --mode all
  fi
done
