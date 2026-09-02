#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash RUN_BASH/run_qwen3vl_aitw_4B.sh
# Optional env vars:
#   MODEL=Qwen3-VL-4B-Instruct
#   WORK_DIR=OUTPUT/aitw_hist4_state_packet_structured_fast_sample_4B
#   CUDA_VISIBLE_DEVICES=1
#   NPROC_PER_NODE=1
#   AITW_SPLIT=test
#   AITW_WITH_NO_HISTORY=0
#   VLM_EVAL_SAMPLE_MODE=task
#   VLM_EVAL_SAMPLE_TASKS=3

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MODEL="${MODEL:-Qwen3-VL-4B-Instruct}"
TAG="${TAG:-hist4_state_packet_fast_decode_sample_num}"
WORK_DIR="${WORK_DIR:-OUTPUT/aitw_${TAG}_4B}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

export AITW_ANN_ROOT="${AITW_ANN_ROOT:-/mnt/storage2/Datasets/aitw_data/aitw_annots}"
export AITW_IMAGE_ROOT="${AITW_IMAGE_ROOT:-/mnt/storage2/Datasets/aitw_data/aitw_images}"
export AITW_SPLIT="${AITW_SPLIT:-test}"
export AITW_HIS_NUM="${AITW_HIS_NUM:-4}"
export AITW_WITH_NO_HISTORY="${AITW_WITH_NO_HISTORY:-0}"
export AITW_SEMANTIC_ACTION_PROMPT="${AITW_SEMANTIC_ACTION_PROMPT:-0}" # auto follows structured fast decode; 1=semantic; 0=numeric
export AITW_DEBUG_HISTORY_PROMPT="${AITW_DEBUG_HISTORY_PROMPT:-1}"

export VLM_TIMING="${VLM_TIMING:-1}"
export VLM_TIMING_VERBOSE="${VLM_TIMING_VERBOSE:-1}"
export VLM_TIMING_SYNC="${VLM_TIMING_SYNC:-1}"
export VLM_STAGE_TIMING="${VLM_STAGE_TIMING:-1}"
export VLM_STAGE_TIMING_DEVICE="${VLM_STAGE_TIMING_DEVICE:-auto}"
export VLM_STAGE_TIMING_SYNC="${VLM_STAGE_TIMING_SYNC:-0}"
export VLM_PRUNE_TIMING="${VLM_PRUNE_TIMING:-1}"
export SEED="${SEED:-42}"
export QWEN3VL_PROFILE_FLOPS="${QWEN3VL_PROFILE_FLOPS:-1}"
export QWEN3VL_RUNTIME_TRACKING="${QWEN3VL_RUNTIME_TRACKING:-1}"

export QWEN3VL_ENABLE_TEMPLATE_PREFILL="${QWEN3VL_ENABLE_TEMPLATE_PREFILL:-0}"
export QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE="${QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE:-1}"
export QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG="${QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG:-1}"
export QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG_POS="${QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG_POS:-0}"
export QWEN3VL_STRUCTURED_FAST_DECODE_ACTION_MAX_TOKENS="${QWEN3VL_STRUCTURED_FAST_DECODE_ACTION_MAX_TOKENS:-16}"
export QWEN3VL_STRUCTURED_FAST_DECODE_AITW_COORD_MAX_TOKENS="${QWEN3VL_STRUCTURED_FAST_DECODE_AITW_COORD_MAX_TOKENS:-24}"
export QWEN3VL_STRUCTURED_FAST_DECODE_AITW_TEXT_MAX_TOKENS="${QWEN3VL_STRUCTURED_FAST_DECODE_AITW_TEXT_MAX_TOKENS:-32}"
export QWEN3VL_STRUCTURED_FAST_DECODE_GUI_DIRECTION_MAX_TOKENS="${QWEN3VL_STRUCTURED_FAST_DECODE_GUI_DIRECTION_MAX_TOKENS:-8}"
export QWEN3VL_STRUCTURED_FAST_DECODE_GUI_COORD_MAX_TOKENS="${QWEN3VL_STRUCTURED_FAST_DECODE_GUI_COORD_MAX_TOKENS:-24}"
export QWEN3VL_STRUCTURED_FAST_DECODE_STATIC_TOKENWISE="${QWEN3VL_STRUCTURED_FAST_DECODE_STATIC_TOKENWISE:-0}"
export QWEN3VL_STRUCTURED_FAST_DECODE_GUI_REQUIRE_ACTION_SEPARATOR="${QWEN3VL_STRUCTURED_FAST_DECODE_GUI_REQUIRE_ACTION_SEPARATOR:-1}"
export QWEN3VL_TEMPLATE_PREFILL_IMPL="${QWEN3VL_TEMPLATE_PREFILL_IMPL:-stateful}"
export QWEN3VL_TEMPLATE_PREFILL_DATASETS="${QWEN3VL_TEMPLATE_PREFILL_DATASETS:-androidcontrol,aitw}"
export QWEN3VL_TEMPLATE_PREFILL_DEBUG="${QWEN3VL_TEMPLATE_PREFILL_DEBUG:-1}"
export QWEN3VL_TEMPLATE_PREFILL_ANDROID_MODE="${QWEN3VL_TEMPLATE_PREFILL_ANDROID_MODE:-action_first_json}"

export QWEN3VL_ENABLE_ROI_PRUNE="${QWEN3VL_ENABLE_ROI_PRUNE:-0}"
export QWEN3VL_ROI_PRUNE_USE_CACHE="${QWEN3VL_ROI_PRUNE_USE_CACHE:-1}"
export QWEN3VL_ROI_PRUNE_PRINT_PER_SAMPLE="${QWEN3VL_ROI_PRUNE_PRINT_PER_SAMPLE:-1}"

export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-task}" # off=full, task=episodes, sample=steps, index=fixed indices
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-3}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"
export VLM_EVAL_SAMPLE_COUNT="${VLM_EVAL_SAMPLE_COUNT:-20}"
export VLM_EVAL_SAMPLE_INDICES="${VLM_EVAL_SAMPLE_INDICES:-}"
export VLM_EVAL_SAMPLE_INDEX_FILE="${VLM_EVAL_SAMPLE_INDEX_FILE:-}"

export AITW_STATE_PACKET_ENABLE="${AITW_STATE_PACKET_ENABLE:-1}"
export AITW_STATE_PACKET_DEBUG="${AITW_STATE_PACKET_DEBUG:-1}"
export AITW_STATE_PACKET_CACHE_DIR="${AITW_STATE_PACKET_CACHE_DIR:-tmp/aitw_state_packet_cache}"
export AITW_STATE_PACKET_PATCH_SIZE="${AITW_STATE_PACKET_PATCH_SIZE:-16}"
export AITW_STATE_PACKET_MERGE_SIZE="${AITW_STATE_PACKET_MERGE_SIZE:-2}"
export AITW_STATE_PACKET_THUMB_LONG_EDGE="${AITW_STATE_PACKET_THUMB_LONG_EDGE:-192}"
export AITW_STATE_PACKET_ROI_LONG_EDGE="${AITW_STATE_PACKET_ROI_LONG_EDGE:-224}"
export AITW_STATE_PACKET_ROI_SHORT_SIDE_RATIO="${AITW_STATE_PACKET_ROI_SHORT_SIDE_RATIO:-0.22}"
export AITW_STATE_PACKET_ROI_MIN_SIDE_PX="${AITW_STATE_PACKET_ROI_MIN_SIDE_PX:-160}"

# AITW prompt/eval uses normalized 0-1000 coordinates. Keep AndroidControl-only
# denormalization disabled for AITW runs.
export QWEN3VL_ANDROID_DENORM_ON_INFER="${QWEN3VL_ANDROID_DENORM_ON_INFER:-0}"
export QWEN3VL_ANDROID_DENORM_BASE="${QWEN3VL_ANDROID_DENORM_BASE:-1000}"

if [[ ! -f "${AITW_ANN_ROOT}/aitw_data_${AITW_SPLIT}.json" ]]; then
  echo "[ERROR] Missing annotation file: ${AITW_ANN_ROOT}/aitw_data_${AITW_SPLIT}.json"
  exit 1
fi
if [[ ! -d "${AITW_IMAGE_ROOT}" ]]; then
  echo "[ERROR] Missing image root: ${AITW_IMAGE_ROOT}"
  exit 1
fi

mkdir -p "${WORK_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="run_output_${TIMESTAMP}_aitw_${TAG}.log"

DS_LIST=(
  "AITW_all"
)

run_one() {
  local dataset_name="$1"
  local port="$2"

  mkdir -p "${WORK_DIR}/${dataset_name}"
  {
    echo "[Run] tag=${TAG}"
    echo "[Run] python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
    echo "[Run] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "[Run] model=${MODEL}"
    echo "[Run] dataset=${dataset_name}"
    echo "[Run] work_dir=${WORK_DIR}"
    echo "[Run] aitw_ann_root=${AITW_ANN_ROOT}"
    echo "[Run] aitw_image_root=${AITW_IMAGE_ROOT}"
    echo "[Run] aitw_split=${AITW_SPLIT}"
    echo "[Run] aitw_his_num=${AITW_HIS_NUM}"
    echo "[Run] aitw_with_no_history=${AITW_WITH_NO_HISTORY}"
    echo "[Run] aitw_semantic_action_prompt=${AITW_SEMANTIC_ACTION_PROMPT}"
    echo "[Run] state_packet_enable=${AITW_STATE_PACKET_ENABLE}"
    echo "[Run] state_packet_debug=${AITW_STATE_PACKET_DEBUG}"
    echo "[Run] thumb_long_edge=${AITW_STATE_PACKET_THUMB_LONG_EDGE}"
    echo "[Run] roi_long_edge=${AITW_STATE_PACKET_ROI_LONG_EDGE}"
    echo "[Run] roi_short_side_ratio=${AITW_STATE_PACKET_ROI_SHORT_SIDE_RATIO}"
    echo "[Run] roi_min_side_px=${AITW_STATE_PACKET_ROI_MIN_SIDE_PX}"
    echo "[Run] flops_profile=${QWEN3VL_PROFILE_FLOPS}"
    echo "[Run] runtime_tracking=${QWEN3VL_RUNTIME_TRACKING}"
    echo "[Run] timing=${VLM_TIMING}"
    echo "[Run] timing_verbose=${VLM_TIMING_VERBOSE}"
    echo "[Run] timing_sync=${VLM_TIMING_SYNC}"
    echo "[Run] stage_timing=${VLM_STAGE_TIMING}"
    echo "[Run] stage_timing_device=${VLM_STAGE_TIMING_DEVICE}"
    echo "[Run] stage_timing_sync=${VLM_STAGE_TIMING_SYNC}"
    echo "[Run] prune_timing=${VLM_PRUNE_TIMING}"
    echo "[Run] template_prefill=${QWEN3VL_ENABLE_TEMPLATE_PREFILL}"
    echo "[Run] structured_fast_decode=${QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE}"
    echo "[Run] structured_fast_decode_debug=${QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG}"
    echo "[Run] structured_fast_decode_debug_pos=${QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG_POS}"
    echo "[Run] structured_fast_decode_action_max_tokens=${QWEN3VL_STRUCTURED_FAST_DECODE_ACTION_MAX_TOKENS}"
    echo "[Run] structured_fast_decode_aitw_coord_max_tokens=${QWEN3VL_STRUCTURED_FAST_DECODE_AITW_COORD_MAX_TOKENS}"
    echo "[Run] structured_fast_decode_aitw_text_max_tokens=${QWEN3VL_STRUCTURED_FAST_DECODE_AITW_TEXT_MAX_TOKENS}"
    echo "[Run] structured_fast_decode_gui_direction_max_tokens=${QWEN3VL_STRUCTURED_FAST_DECODE_GUI_DIRECTION_MAX_TOKENS}"
    echo "[Run] structured_fast_decode_gui_coord_max_tokens=${QWEN3VL_STRUCTURED_FAST_DECODE_GUI_COORD_MAX_TOKENS}"
    echo "[Run] structured_fast_decode_static_tokenwise=${QWEN3VL_STRUCTURED_FAST_DECODE_STATIC_TOKENWISE}"
    echo "[Run] structured_fast_decode_gui_require_action_separator=${QWEN3VL_STRUCTURED_FAST_DECODE_GUI_REQUIRE_ACTION_SEPARATOR}"
    echo "[Run] template_prefill_impl=${QWEN3VL_TEMPLATE_PREFILL_IMPL}"
    echo "[Run] template_prefill_datasets=${QWEN3VL_TEMPLATE_PREFILL_DATASETS}"
    echo "[Run] template_prefill_android_mode=${QWEN3VL_TEMPLATE_PREFILL_ANDROID_MODE}"
    echo "[Run] template_prefill_debug=${QWEN3VL_TEMPLATE_PREFILL_DEBUG}"
    echo "[Run] roi_prune_enable=${QWEN3VL_ENABLE_ROI_PRUNE}"
    echo "[Run] roi_prune_use_cache=${QWEN3VL_ROI_PRUNE_USE_CACHE}"
    echo "[Run] roi_prune_print_per_sample=${QWEN3VL_ROI_PRUNE_PRINT_PER_SAMPLE}"
    echo "[Run] sample_mode=${VLM_EVAL_SAMPLE_MODE}"
    echo "[Run] sample_tasks=${VLM_EVAL_SAMPLE_TASKS}"
    echo "[Run] sample_count=${VLM_EVAL_SAMPLE_COUNT}"
    echo "[Run] sample_seed=${VLM_EVAL_SAMPLE_SEED}"
    echo "[Run] sample_indices=${VLM_EVAL_SAMPLE_INDICES}"
    echo "[Run] sample_index_file=${VLM_EVAL_SAMPLE_INDEX_FILE}"
  } | tee -a "${LOG_FILE}"

  local reuse_args=()
  if [[ "${VLM_RUN_REUSE:-0}" == "1" ]]; then
    reuse_args+=(--reuse)
  fi

  "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" --master_port="${port}" run.py \
      --data "${dataset_name}" \
      --model "${MODEL}" \
      --work-dir "${WORK_DIR}/${dataset_name}" \
      --mode all \
      "${reuse_args[@]}" 2>&1 | tee -a "${LOG_FILE}"
}

port_base=29704
idx=0
for ds in "${DS_LIST[@]}"; do
  run_one "${ds}" "$((port_base + idx))"
  idx=$((idx + 1))
done
