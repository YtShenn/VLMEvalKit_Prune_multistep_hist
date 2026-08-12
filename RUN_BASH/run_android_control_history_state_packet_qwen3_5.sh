#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-1}"

export ANDROID_CONTROL_CURATED_EVAL_MODE="${ANDROID_CONTROL_CURATED_EVAL_MODE:-official}"
export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images}"
export ANDROID_CONTROL_DEBUG_HISTORY_PROMPT="${ANDROID_CONTROL_DEBUG_HISTORY_PROMPT:-1}"
export ANDROID_CONTROL_SEQUENTIAL_ORDER="${ANDROID_CONTROL_SEQUENTIAL_ORDER:-1}"

export QWEN3VL_ANDROID_DENORM_ON_INFER="${QWEN3VL_ANDROID_DENORM_ON_INFER:-1}"
export QWEN3VL_ANDROID_DENORM_BASE="${QWEN3VL_ANDROID_DENORM_BASE:-1000}"

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

export QWEN3VL_ENABLE_TEMPLATE_PREFILL="${QWEN3VL_ENABLE_TEMPLATE_PREFILL:-1}"
export QWEN3VL_TEMPLATE_PREFILL_IMPL="${QWEN3VL_TEMPLATE_PREFILL_IMPL:-stateful}" #single_generate
export QWEN3VL_TEMPLATE_PREFILL_DATASETS="${QWEN3VL_TEMPLATE_PREFILL_DATASETS:-androidcontrol}"
export QWEN3VL_TEMPLATE_PREFILL_DEBUG="${QWEN3VL_TEMPLATE_PREFILL_DEBUG:-1}"
export QWEN3VL_TEMPLATE_PREFILL_ANDROID_MODE="${QWEN3VL_TEMPLATE_PREFILL_ANDROID_MODE:-action_first_json}" #bbox_first_json

export QWEN3VL_ENABLE_ROI_PRUNE=0
export QWEN3VL_ROI_PRUNE_USE_CACHE="${QWEN3VL_ROI_PRUNE_USE_CACHE:-1}"
export QWEN3VL_ROI_PRUNE_PRINT_PER_SAMPLE="${QWEN3VL_ROI_PRUNE_PRINT_PER_SAMPLE:-1}"

export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-task}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-10}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"

export ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS="${ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS:-1}"
export ANDROID_CONTROL_MAX_HISTORY_IMAGES="${ANDROID_CONTROL_MAX_HISTORY_IMAGES:-4}"
export ANDROID_CONTROL_HISTORY_KEEP_SYSTEM_PROMPT="${ANDROID_CONTROL_HISTORY_KEEP_SYSTEM_PROMPT:-1}"

export ANDROID_CONTROL_STATE_PACKET_ENABLE="1" #${ANDROID_CONTROL_STATE_PACKET_ENABLE:-1}"
export ANDROID_CONTROL_STATE_PACKET_DEBUG="${ANDROID_CONTROL_STATE_PACKET_DEBUG:-1}"
export ANDROID_CONTROL_STATE_PACKET_CACHE_DIR="${ANDROID_CONTROL_STATE_PACKET_CACHE_DIR:-/tmp/android_control_state_packet_cache}"
export ANDROID_CONTROL_STATE_PACKET_PATCH_SIZE="${ANDROID_CONTROL_STATE_PACKET_PATCH_SIZE:-16}"
export ANDROID_CONTROL_STATE_PACKET_MERGE_SIZE="${ANDROID_CONTROL_STATE_PACKET_MERGE_SIZE:-2}"
# AndroidControl keeps the current screenshot as a full image, so we bias the
# history packet a bit more aggressively toward speed than GUI Odyssey does.
export ANDROID_CONTROL_STATE_PACKET_THUMB_LONG_EDGE="${ANDROID_CONTROL_STATE_PACKET_THUMB_LONG_EDGE:-192}"  #256
export ANDROID_CONTROL_STATE_PACKET_ROI_LONG_EDGE="${ANDROID_CONTROL_STATE_PACKET_ROI_LONG_EDGE:-224}"  #256
export ANDROID_CONTROL_STATE_PACKET_ROI_SHORT_SIDE_RATIO="${ANDROID_CONTROL_STATE_PACKET_ROI_SHORT_SIDE_RATIO:-0.22}"   #0.28
export ANDROID_CONTROL_STATE_PACKET_ROI_MIN_SIDE_PX="${ANDROID_CONTROL_STATE_PACKET_ROI_MIN_SIDE_PX:-160}"  #224

MODEL="${MODEL:-Qwen3-VL-4B-Instruct}"
DATASET_LIST=(
    # "AndroidControl_Curated_High_Point"
    "AndroidControl_Curated_High_Task_Improved"
)
PYTHON_BIN="${PYTHON_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/torchrun}"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-hist4_keep_system_prompt_state_packet_new_template_decode_official}"
WORK_DIR="${WORK_DIR:-OUTPUT/ablation_android_control_${TAG}_node3_0811(only_HTI)_ratio2}"
LOG_FILE="run_output_${TS}_android_control_${TAG}.log"

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
    echo "[Run] eval_mode=${ANDROID_CONTROL_CURATED_EVAL_MODE}"
    echo "[Run] use_history=${ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS}"
    echo "[Run] max_history=${ANDROID_CONTROL_MAX_HISTORY_IMAGES}"
    echo "[Run] keep_system_prompt=${ANDROID_CONTROL_HISTORY_KEEP_SYSTEM_PROMPT}"
    echo "[Run] state_packet_enable=${ANDROID_CONTROL_STATE_PACKET_ENABLE}"
    echo "[Run] state_packet_debug=${ANDROID_CONTROL_STATE_PACKET_DEBUG}"
    echo "[Run] thumb_long_edge=${ANDROID_CONTROL_STATE_PACKET_THUMB_LONG_EDGE}"
    echo "[Run] roi_long_edge=${ANDROID_CONTROL_STATE_PACKET_ROI_LONG_EDGE}"
    echo "[Run] roi_short_side_ratio=${ANDROID_CONTROL_STATE_PACKET_ROI_SHORT_SIDE_RATIO}"
    echo "[Run] roi_min_side_px=${ANDROID_CONTROL_STATE_PACKET_ROI_MIN_SIDE_PX}"
    echo "[Run] flops_profile=${QWEN3VL_PROFILE_FLOPS}"
    echo "[Run] template_prefill=${QWEN3VL_ENABLE_TEMPLATE_PREFILL}"
    echo "[Run] template_prefill_impl=${QWEN3VL_TEMPLATE_PREFILL_IMPL}"
    echo "[Run] template_prefill_datasets=${QWEN3VL_TEMPLATE_PREFILL_DATASETS}"
    echo "[Run] template_prefill_android_mode=${QWEN3VL_TEMPLATE_PREFILL_ANDROID_MODE}"
    echo "[Run] sample_mode=${VLM_EVAL_SAMPLE_MODE}"
    echo "[Run] sample_tasks=${VLM_EVAL_SAMPLE_TASKS}"
    echo "[Run] sample_seed=${VLM_EVAL_SAMPLE_SEED}"
  } | tee -a "${LOG_FILE}"

  local reuse_args=()
  if [[ "${VLM_RUN_REUSE:-0}" == "1" ]]; then
    reuse_args+=(--reuse)
  fi

  "${TORCHRUN_BIN}" --standalone --nproc_per_node=1 --master_port="${port}" run.py \
    --data "${dataset_name}" \
    --model "${MODEL}" \
    --work-dir "${WORK_DIR}/${dataset_name}" \
    --mode all \
    "${reuse_args[@]}" 2>&1 | tee -a "${LOG_FILE}"
}

port_base=29664
idx=0
for dataset_name in "${DATASET_LIST[@]}"; do
  run_one "${dataset_name}" "$((port_base + idx))"
  idx=$((idx + 1))
done
