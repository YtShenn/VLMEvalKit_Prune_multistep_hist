#!/usr/bin/env bash
set -euo pipefail

# Faithful FastV with four old->new history screenshots plus the current one.
# Example: DATASETS='AndroidControl_Curated_High_Task_Improved' CUDA_VISIBLE_DEVICES=0 bash RUN_BASH/run_qwen3vl_fastv_4B.sh
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export SEED="${SEED:-42}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
MODEL="${MODEL:-Qwen3-VL-4B-Instruct-FastV}"
# DATASETS="${DATASETS:-AndroidControl_Curated_High_Task_Improved}"
# DATASETS="${DATASETS:-AITW_all}"
DATASETS="${DATASETS:-Mind2Web_test_task}"
# DATASETS="${DATASETS:-GUIOdyssey_high_task_split}"
WORK_DIR="${WORK_DIR:-OUTPUT_FASTV/4B_hist4_k${FASTV_K:-2}_r${FASTV_R:-0.5}_flops2}"

# Official FastV controls. R is the discard ratio over all five images.
export FASTV_ENABLED="${FASTV_ENABLED:-1}"
export FASTV_K="${FASTV_K:-2}"
export FASTV_R="${FASTV_R:-0.4}"
# Per-sample FastV JSON diagnostics are noisy in full evaluation. Enable only
# when investigating a run: FASTV_DEBUG=1 bash ...
export FASTV_DEBUG="${FASTV_DEBUG:-0}"

# Four-history-step prompt construction.
export GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS="${GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS:-1}"
export GUI_ODYSSEY_MAX_HISTORY_IMAGES="${GUI_ODYSSEY_MAX_HISTORY_IMAGES:-4}"
export ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS="${ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS:-1}"
export ANDROID_CONTROL_MAX_HISTORY_IMAGES="${ANDROID_CONTROL_MAX_HISTORY_IMAGES:-4}"
export AITW_HIS_NUM="${AITW_HIS_NUM:-4}"
export MIND2WEB_HIS_NUM="${MIND2WEB_HIS_NUM:-4}"

# FastV is isolated from the project's other accelerators.
export QWEN3VL_ENABLE_ATTN_PRUNE=0
export QWEN3VL_ENABLE_ROI_PRUNE=0
export QWEN3VL_ENABLE_TEMPLATE_PREFILL=0
export QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE=0
export AITW_STATE_PACKET_ENABLE=0
export MIND2WEB_STATE_PACKET_ENABLE=0
export QWEN3VL_ANDROID_DENORM_ON_INFER="${QWEN3VL_ANDROID_DENORM_ON_INFER:-1}"
export QWEN3VL_ANDROID_DENORM_BASE="${QWEN3VL_ANDROID_DENORM_BASE:-1000}"

# Runtime timing and FLOPs profiling. Set PROFILE_FLOPS=0 for throughput-only.
export VLM_TIMING="${VLM_TIMING:-1}"
export VLM_TIMING_VERBOSE="${VLM_TIMING_VERBOSE:-1}"
export VLM_TIMING_SYNC="${VLM_TIMING_SYNC:-0}"
export VLM_STAGE_TIMING="${VLM_STAGE_TIMING:-1}"
export VLM_STAGE_TIMING_DEVICE="${VLM_STAGE_TIMING_DEVICE:-auto}"
export VLM_STAGE_TIMING_SYNC="${VLM_STAGE_TIMING_SYNC:-0}"
export QWEN3VL_RUNTIME_TRACKING="${QWEN3VL_RUNTIME_TRACKING:-1}"
export QWEN3VL_PROFILE_FLOPS="${QWEN3VL_PROFILE_FLOPS:-1}"

# Default small task subset; set VLM_EVAL_SAMPLE_MODE=off for a full run.
export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-off}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-5}"
export VLM_EVAL_SAMPLE_COUNT="${VLM_EVAL_SAMPLE_COUNT:-20}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"

# Dataset directories. Every value can be overridden in the launching shell.
export AITW_ANN_ROOT="${AITW_ANN_ROOT:-/public/ytshen/aitw_data/aitw_annots}"
export AITW_IMAGE_ROOT="${AITW_IMAGE_ROOT:-/public/ytshen/aitw_data/aitw_images}"
export AITW_SPLIT="${AITW_SPLIT:-test}"
export AITW_WITH_NO_HISTORY="${AITW_WITH_NO_HISTORY:-0}"
export AITW_SEMANTIC_ACTION_PROMPT="${AITW_SEMANTIC_ACTION_PROMPT:-0}"
export MIND2WEB_DATA_ROOT="${MIND2WEB_DATA_ROOT:-/public/ytshen/Mind2Web}"
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
export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/public/ytshen/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/public/ytshen/AndroidControl_Curated/images}"

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
STAMP="$(date +%Y%m%d_%H%M%S)"
for dataset in ${DATASETS}; do
  require_dataset_files "${dataset}"
  output_dir="${WORK_DIR}/${dataset}"
  log_file="${WORK_DIR}/run_output_${STAMP}_${dataset}_fastv.log"
  mkdir -p "${output_dir}"
  {
    echo "[FastV] model=${MODEL} dataset=${dataset} output=${output_dir}"
    echo "[FastV] history_frames=4 k=${FASTV_K} discard_ratio=${FASTV_R} enabled=${FASTV_ENABLED}"
    echo "[Timing] stage=${VLM_STAGE_TIMING} runtime_tracking=${QWEN3VL_RUNTIME_TRACKING} profile_flops=${QWEN3VL_PROFILE_FLOPS} sync=${VLM_TIMING_SYNC}"
    echo "[Sample] mode=${VLM_EVAL_SAMPLE_MODE} tasks=${VLM_EVAL_SAMPLE_TASKS} count=${VLM_EVAL_SAMPLE_COUNT}"
  } | tee -a "${log_file}"
  if [[ "${NPROC_PER_NODE}" == "1" ]]; then
    "${PYTHON_BIN}" run.py --data "${dataset}" --model "${MODEL}" --work-dir "${output_dir}" --mode all 2>&1 | tee -a "${log_file}"
  else
    "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" run.py --data "${dataset}" --model "${MODEL}" --work-dir "${output_dir}" --mode all 2>&1 | tee -a "${log_file}"
  fi
done
