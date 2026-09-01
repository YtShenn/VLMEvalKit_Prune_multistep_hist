#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6}"
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
export VLM_STAGE_TIMING="${VLM_STAGE_TIMING:-0}"
export VLM_STAGE_TIMING_DEVICE="${VLM_STAGE_TIMING_DEVICE:-auto}"
export VLM_STAGE_TIMING_SYNC="${VLM_STAGE_TIMING_SYNC:-0}"
export VLM_PRUNE_TIMING="${VLM_PRUNE_TIMING:-0}"
export SEED="${SEED:-42}"
export QWEN3VL_PROFILE_FLOPS="${QWEN3VL_PROFILE_FLOPS:-0}"
export QWEN3VL_RUNTIME_TRACKING="${QWEN3VL_RUNTIME_TRACKING:-0}"

export QWEN3VL_ENABLE_TEMPLATE_PREFILL="${QWEN3VL_ENABLE_TEMPLATE_PREFILL:-0}"
export QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE="${QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE:-1}"
export QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG="${QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG:-1}"
export QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG_POS="${QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG_POS:-0}"
export QWEN3VL_TEMPLATE_PREFILL_IMPL="${QWEN3VL_TEMPLATE_PREFILL_IMPL:-stateful}"
export QWEN3VL_TEMPLATE_PREFILL_DATASETS="${QWEN3VL_TEMPLATE_PREFILL_DATASETS:-androidcontrol}"
export QWEN3VL_TEMPLATE_PREFILL_DEBUG="${QWEN3VL_TEMPLATE_PREFILL_DEBUG:-1}"
export QWEN3VL_TEMPLATE_PREFILL_ANDROID_MODE="${QWEN3VL_TEMPLATE_PREFILL_ANDROID_MODE:-action_first_json}"

export QWEN3VL_ENABLE_ROI_PRUNE=0
export QWEN3VL_ENABLE_ATTN_PRUNE="${QWEN3VL_ENABLE_ATTN_PRUNE:-0}"
export QWEN3VL_ATTN_PRUNE_LAYERS="${QWEN3VL_ATTN_PRUNE_LAYERS:-}"
export QWEN3VL_ATTN_PRUNE_KEEP_RATIO="${QWEN3VL_ATTN_PRUNE_KEEP_RATIO:-1.0}"
export QWEN3VL_ATTN_PRUNE_USE_CACHE="${QWEN3VL_ATTN_PRUNE_USE_CACHE:-1}"
export QWEN3VL_ATTN_PRUNE_STRICT_QUERY_MARKER="${QWEN3VL_ATTN_PRUNE_STRICT_QUERY_MARKER:-1}"
export QWEN3VL_ATTN_PRUNE_PRINT_QUERY_TOKENS="${QWEN3VL_ATTN_PRUNE_PRINT_QUERY_TOKENS:-0}"
export QWEN3VL_ATTN_PRUNE_SIDE_COMPUTE="${QWEN3VL_ATTN_PRUNE_SIDE_COMPUTE:-1}"
export QWEN3VL_ATTN_PRUNE_DEBUG="${QWEN3VL_ATTN_PRUNE_DEBUG:-1}"

# Confidence observer only. It side-computes instruction-to-current-frame
# attention on selected prefill layers and never drops tokens.
export QWEN3VL_ATTN_CONF_ENABLE="${QWEN3VL_ATTN_CONF_ENABLE:-1}"
export QWEN3VL_ATTN_CONF_LAYERS="${QWEN3VL_ATTN_CONF_LAYERS:-0,1,2,3}"
export QWEN3VL_ATTN_CONF_ANALYZE="${QWEN3VL_ATTN_CONF_ANALYZE:-1}"
export QWEN3VL_ATTN_CONF_ANALYSIS_METRICS="${QWEN3VL_ATTN_CONF_ANALYSIS_METRICS:-confidence,uncertainty,top10_mass,top20_mass,gini,max_mean_ratio}"
export QWEN3VL_ATTN_CONF_ANALYSIS_BINS="${QWEN3VL_ATTN_CONF_ANALYSIS_BINS:-4}"

export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-off}" #index
export VLM_EVAL_SAMPLE_INDEX_FILE="${VLM_EVAL_SAMPLE_INDEX_FILE:-OUTPUT_0/attn_prune_debug/android_control_history_state_packet_step_instruct/selected_indices.txt}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-10}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"

export ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS="${ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS:-1}"
export ANDROID_CONTROL_MAX_HISTORY_IMAGES="${ANDROID_CONTROL_MAX_HISTORY_IMAGES:-4}"
export ANDROID_CONTROL_HISTORY_KEEP_SYSTEM_PROMPT="${ANDROID_CONTROL_HISTORY_KEEP_SYSTEM_PROMPT:-1}"

export ANDROID_CONTROL_STATE_PACKET_ENABLE="${ANDROID_CONTROL_STATE_PACKET_ENABLE:-1}"
export ANDROID_CONTROL_STATE_PACKET_DEBUG="${ANDROID_CONTROL_STATE_PACKET_DEBUG:-1}"
export ANDROID_CONTROL_STATE_PACKET_CACHE_DIR="${ANDROID_CONTROL_STATE_PACKET_CACHE_DIR:-/tmp/android_control_state_packet_cache}"
export ANDROID_CONTROL_STATE_PACKET_PATCH_SIZE="${ANDROID_CONTROL_STATE_PACKET_PATCH_SIZE:-16}"
export ANDROID_CONTROL_STATE_PACKET_MERGE_SIZE="${ANDROID_CONTROL_STATE_PACKET_MERGE_SIZE:-2}"
export ANDROID_CONTROL_STATE_PACKET_THUMB_LONG_EDGE="${ANDROID_CONTROL_STATE_PACKET_THUMB_LONG_EDGE:-192}"
export ANDROID_CONTROL_STATE_PACKET_ROI_LONG_EDGE="${ANDROID_CONTROL_STATE_PACKET_ROI_LONG_EDGE:-224}"
export ANDROID_CONTROL_STATE_PACKET_ROI_SHORT_SIDE_RATIO="${ANDROID_CONTROL_STATE_PACKET_ROI_SHORT_SIDE_RATIO:-0.22}"
export ANDROID_CONTROL_STATE_PACKET_ROI_MIN_SIDE_PX="${ANDROID_CONTROL_STATE_PACKET_ROI_MIN_SIDE_PX:-160}"

MODEL="${MODEL:-Qwen3-VL-4B-Instruct-AttnPrune}"
DATASET_LIST=(
    "AndroidControl_Curated_High_Task_Improved"
)
PYTHON_BIN="${PYTHON_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-hist4_keep_system_prompt_state_packet_attn_conf_observe_layers${QWEN3VL_ATTN_CONF_LAYERS//,/}_0830_high_all}"
WORK_DIR="${WORK_DIR:-OUTPUT/android_control_${TAG}}"
LOG_FILE="run_output_${TS}_android_control_${TAG}.log"

run_one() {
  local dataset_name="$1"
  local port="$2"
  local dataset_work_dir="${WORK_DIR}/${dataset_name}"
  local conf_out_dir="${QWEN3VL_ATTN_CONF_OUT_DIR:-${dataset_work_dir}/attn_confidence}"
  local analysis_out_dir="${QWEN3VL_ATTN_CONF_ANALYSIS_OUT_DIR:-${dataset_work_dir}/attn_confidence_analysis}"

  export QWEN3VL_ATTN_CONF_OUT_DIR="${conf_out_dir}"
  export QWEN3VL_ATTN_PRUNE_OUT_DIR="${QWEN3VL_ATTN_PRUNE_OUT_DIR:-${conf_out_dir}}"

  mkdir -p "${dataset_work_dir}" "${conf_out_dir}" "${analysis_out_dir}"
  {
    echo "[Run] tag=${TAG}"
    echo "[Run] python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
    echo "[Run] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "[Run] nproc_per_node=${NPROC_PER_NODE}"
    echo "[Run] model=${MODEL}"
    echo "[Run] dataset=${dataset_name}"
    echo "[Run] work_dir=${WORK_DIR}"
    echo "[Run] state_packet_enable=${ANDROID_CONTROL_STATE_PACKET_ENABLE}"
    echo "[Run] structured_fast_decode=${QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE}"
    echo "[Run] attn_prune_enable=${QWEN3VL_ENABLE_ATTN_PRUNE}"
    echo "[Run] attn_prune_layers=${QWEN3VL_ATTN_PRUNE_LAYERS}"
    echo "[Run] attn_prune_keep_ratio=${QWEN3VL_ATTN_PRUNE_KEEP_RATIO}"
    echo "[Run] attn_conf_enable=${QWEN3VL_ATTN_CONF_ENABLE}"
    echo "[Run] attn_conf_layers=${QWEN3VL_ATTN_CONF_LAYERS}"
    echo "[Run] attn_conf_out_dir=${QWEN3VL_ATTN_CONF_OUT_DIR}"
    echo "[Run] attn_conf_analyze=${QWEN3VL_ATTN_CONF_ANALYZE}"
    echo "[Run] sample_mode=${VLM_EVAL_SAMPLE_MODE}"
    echo "[Run] sample_index_file=${VLM_EVAL_SAMPLE_INDEX_FILE:-}"
    echo "[Run] sample_tasks=${VLM_EVAL_SAMPLE_TASKS}"
  } | tee -a "${LOG_FILE}"

  local reuse_args=()
  if [[ "${VLM_RUN_REUSE:-0}" == "1" ]]; then
    reuse_args+=(--reuse)
  fi

  "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" --master_port="${port}" run.py \
    --data "${dataset_name}" \
    --model "${MODEL}" \
    --work-dir "${dataset_work_dir}" \
    --mode all \
    "${reuse_args[@]}" 2>&1 | tee -a "${LOG_FILE}"

  if [[ "${QWEN3VL_ATTN_CONF_ANALYZE}" == "1" ]]; then
    local conf_jsonl="${conf_out_dir}/attn_confidence_records.jsonl"
    if [[ ! -s "${conf_jsonl}" ]]; then
      echo "[AttnConfidenceAnalysis] skip: missing ${conf_jsonl}" | tee -a "${LOG_FILE}"
      return
    fi
    local detail_json
    detail_json="$(find "${dataset_work_dir}" -name '*_android_control_detail*.json' -type f -print | sort | tail -n 1)"
    if [[ -z "${detail_json}" ]]; then
      echo "[AttnConfidenceAnalysis] skip: no *_android_control_detail*.json under ${dataset_work_dir}" | tee -a "${LOG_FILE}"
      return
    fi
    "${PYTHON_BIN}" utils/analyze_attn_confidence_success.py \
      --confidence-jsonl "${conf_jsonl}" \
      --detail-json "${detail_json}" \
      --work-dir "${dataset_work_dir}" \
      --out-dir "${analysis_out_dir}" \
      --metrics "${QWEN3VL_ATTN_CONF_ANALYSIS_METRICS}" \
      --bins "${QWEN3VL_ATTN_CONF_ANALYSIS_BINS}" 2>&1 | tee -a "${LOG_FILE}"
  fi
}

port_base=29674
idx=0
for dataset_name in "${DATASET_LIST[@]}"; do
  run_one "${dataset_name}" "$((port_base + idx))"
  idx=$((idx + 1))
done
