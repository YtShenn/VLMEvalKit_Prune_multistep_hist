#!/usr/bin/env bash
set -euo pipefail

# Training-free Qwen3VL-PruMerge-inspired comparison. All defaults may be
# overridden inline, e.g. DATASETS='MMBench_DEV_EN' VLM_EVAL_SAMPLE_MODE=off \
# CUDA_VISIBLE_DEVICES=0 bash RUN_BASH/run_qwen3vl_prumerge_4B.sh
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export SEED="${SEED:-42}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MODEL="${MODEL:-Qwen3-VL-4B-Instruct-PruMerge}"
DATASETS="${DATASETS:-AndroidControl_Curated_High_Task_Improved}"
WORK_DIR="${WORK_DIR:-OUTPUT_PRUMERGE/4B_${QWEN3VL_PRUMERGE_VARIANT:-prumerge}_hist4}"

# PruMerge controls. The Python wrapper is disabled by default, while this
# explicit comparison runner enables it.
export QWEN3VL_PRUMERGE_ENABLED="${QWEN3VL_PRUMERGE_ENABLED:-1}"
export QWEN3VL_PRUMERGE_VARIANT="${QWEN3VL_PRUMERGE_VARIANT:-prumerge}"
export QWEN3VL_PRUMERGE_MIN_KEEP_TOKENS="${QWEN3VL_PRUMERGE_MIN_KEEP_TOKENS:-4}"
export QWEN3VL_PRUMERGE_MAX_KEEP_TOKENS="${QWEN3VL_PRUMERGE_MAX_KEEP_TOKENS:-128}"
export QWEN3VL_PRUMERGE_IQR_MULTIPLIER="${QWEN3VL_PRUMERGE_IQR_MULTIPLIER:-1.5}"
export QWEN3VL_PRUMERGE_SIMILARITY_TEMPERATURE="${QWEN3VL_PRUMERGE_SIMILARITY_TEMPERATURE:-1.0}"
export QWEN3VL_PRUMERGE_MAX_HISTORY_STEPS="${QWEN3VL_PRUMERGE_MAX_HISTORY_STEPS:-4}"
export QWEN3VL_PRUMERGE_SCOPE="${QWEN3VL_PRUMERGE_SCOPE:-all}" # all/history/current
export QWEN3VL_PRUMERGE_DEBUG="${QWEN3VL_PRUMERGE_DEBUG:-0}"

# Unified current + <=4-history protocol for existing multi-step adapters.
export GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS="${GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS:-1}"
export GUI_ODYSSEY_MAX_HISTORY_IMAGES="${GUI_ODYSSEY_MAX_HISTORY_IMAGES:-4}"
export ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS="${ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS:-1}"
export ANDROID_CONTROL_MAX_HISTORY_IMAGES="${ANDROID_CONTROL_MAX_HISTORY_IMAGES:-4}"
export AITW_HIS_NUM="${AITW_HIS_NUM:-4}"
export MIND2WEB_HIS_NUM="${MIND2WEB_HIS_NUM:-4}"

# Keep this a pure PruMerge baseline.
export QWEN3VL_ENABLE_ATTN_PRUNE=0
export QWEN3VL_ENABLE_ROI_PRUNE=0
export QWEN3VL_ENABLE_TEMPLATE_PREFILL=0
export QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE=0
export AITW_STATE_PACKET_ENABLE=0
export MIND2WEB_STATE_PACKET_ENABLE=0
export QWEN3VL_ANDROID_DENORM_ON_INFER="${QWEN3VL_ANDROID_DENORM_ON_INFER:-1}"
export QWEN3VL_ANDROID_DENORM_BASE="${QWEN3VL_ANDROID_DENORM_BASE:-1000}"

# Cache, timing and FLOPs. Runtime tracking is compression-aware; stage timing
# separates vision and language stages. Set PROFILE_FLOPS=0 for throughput only.
export VLM_TIMING="${VLM_TIMING:-1}"
export VLM_TIMING_VERBOSE="${VLM_TIMING_VERBOSE:-1}"
export VLM_TIMING_SYNC="${VLM_TIMING_SYNC:-0}"
export VLM_STAGE_TIMING="${VLM_STAGE_TIMING:-1}"
export VLM_STAGE_TIMING_DEVICE="${VLM_STAGE_TIMING_DEVICE:-auto}"
export VLM_STAGE_TIMING_SYNC="${VLM_STAGE_TIMING_SYNC:-0}"
export VLM_PRUNE_TIMING="${VLM_PRUNE_TIMING:-1}"
export QWEN3VL_RUNTIME_TRACKING="${QWEN3VL_RUNTIME_TRACKING:-1}"
export QWEN3VL_PROFILE_FLOPS="${QWEN3VL_PROFILE_FLOPS:-1}"

# Task-level sampling defaults. Set VLM_EVAL_SAMPLE_MODE=off for a full run.
export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-task}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-5}"
export VLM_EVAL_SAMPLE_COUNT="${VLM_EVAL_SAMPLE_COUNT:-20}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"

# Dataset locations for multi-step GUI datasets. Ordinary VLMEvalKit datasets
# use their normal configuration/.env paths and are deliberately not blocked.
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
    *) echo "[PruMerge] dataset=${dataset}: using standard VLMEvalKit dataset configuration"; return 0 ;;
  esac || { echo "[ERROR] Missing dataset files for ${dataset}"; return 1; }
}

mkdir -p "${WORK_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
for dataset in ${DATASETS}; do
  require_dataset_files "${dataset}"
  output_dir="${WORK_DIR}/${dataset}"
  log_file="${WORK_DIR}/run_output_${STAMP}_${dataset}_prumerge.log"
  mkdir -p "${output_dir}"
  {
    echo "[PruMerge] model=${MODEL} dataset=${dataset} output=${output_dir}"
    echo "[PruMerge] enabled=${QWEN3VL_PRUMERGE_ENABLED} variant=${QWEN3VL_PRUMERGE_VARIANT} scope=${QWEN3VL_PRUMERGE_SCOPE} min_keep=${QWEN3VL_PRUMERGE_MIN_KEEP_TOKENS} max_keep=${QWEN3VL_PRUMERGE_MAX_KEEP_TOKENS} iqr=${QWEN3VL_PRUMERGE_IQR_MULTIPLIER} history=4"
    echo "[Timing] timing=${VLM_TIMING} stage=${VLM_STAGE_TIMING} prune=${VLM_PRUNE_TIMING} runtime=${QWEN3VL_RUNTIME_TRACKING} flops=${QWEN3VL_PROFILE_FLOPS} sync=${VLM_TIMING_SYNC}"
    echo "[Sample] mode=${VLM_EVAL_SAMPLE_MODE} tasks=${VLM_EVAL_SAMPLE_TASKS} count=${VLM_EVAL_SAMPLE_COUNT} seed=${VLM_EVAL_SAMPLE_SEED}"
  } | tee -a "${log_file}"
  if [[ "${NPROC_PER_NODE}" == "1" ]]; then
    "${PYTHON_BIN}" run.py --data "${dataset}" --model "${MODEL}" --work-dir "${output_dir}" --mode all 2>&1 | tee -a "${log_file}"
  else
    "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" run.py --data "${dataset}" --model "${MODEL}" --work-dir "${output_dir}" --mode all 2>&1 | tee -a "${log_file}"
  fi
done
