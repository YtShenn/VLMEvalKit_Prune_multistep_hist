#!/usr/bin/env bash
# Run Mind2Web history-modality ablations: legacy / none / text / thumbnail / roi.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export MIND2WEB_DATA_ROOT="${MIND2WEB_DATA_ROOT:-/public/ytshen/Mind2Web}"
export MIND2WEB_ANN_ROOT="${MIND2WEB_ANN_ROOT:-${MIND2WEB_DATA_ROOT}/mind2web_annots}"
if [[ -z "${MIND2WEB_IMAGE_ROOT:-}" && -d "${MIND2WEB_DATA_ROOT}/mind2web_images/ming2web_images" ]]; then
  export MIND2WEB_IMAGE_ROOT="${MIND2WEB_DATA_ROOT}/mind2web_images/ming2web_images"
else
  export MIND2WEB_IMAGE_ROOT="${MIND2WEB_IMAGE_ROOT:-${MIND2WEB_DATA_ROOT}/mind2web_images}"
fi
export MIND2WEB_HIS_NUM="${MIND2WEB_HIS_NUM:-4}"
export MIND2WEB_WITH_NO_HISTORY=0
export MIND2WEB_STRICT_OUTPUT_PROMPT="${MIND2WEB_STRICT_OUTPUT_PROMPT:-1}"
export MIND2WEB_STATE_PACKET_DEBUG="${MIND2WEB_STATE_PACKET_DEBUG:-0}"
export MIND2WEB_STATE_PACKET_CACHE_DIR="${MIND2WEB_STATE_PACKET_CACHE_DIR:-/tmp/mind2web_state_packet_cache}"
export MIND2WEB_STATE_PACKET_PATCH_SIZE="${MIND2WEB_STATE_PACKET_PATCH_SIZE:-16}"
export MIND2WEB_STATE_PACKET_MERGE_SIZE="${MIND2WEB_STATE_PACKET_MERGE_SIZE:-2}"
export MIND2WEB_STATE_PACKET_THUMB_LONG_EDGE="${MIND2WEB_STATE_PACKET_THUMB_LONG_EDGE:-192}"
export MIND2WEB_STATE_PACKET_ROI_LONG_EDGE="${MIND2WEB_STATE_PACKET_ROI_LONG_EDGE:-224}"
export MIND2WEB_STATE_PACKET_ROI_SHORT_SIDE_RATIO="${MIND2WEB_STATE_PACKET_ROI_SHORT_SIDE_RATIO:-0.22}"
export MIND2WEB_STATE_PACKET_ROI_MIN_SIDE_PX="${MIND2WEB_STATE_PACKET_ROI_MIN_SIDE_PX:-160}"

export QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE="${QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE:-0}"
export QWEN3VL_MAX_NEW_TOKENS="${QWEN3VL_MAX_NEW_TOKENS:-128}"
export QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_ACTION_MAX_TOKENS="${QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_ACTION_MAX_TOKENS:-4}"
export QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_COORD_MAX_TOKENS="${QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_COORD_MAX_TOKENS:-24}"
export QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_VALUE_MAX_TOKENS="${QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_VALUE_MAX_TOKENS:-32}"
export VLM_TIMING="${VLM_TIMING:-1}"
export VLM_TIMING_VERBOSE="${VLM_TIMING_VERBOSE:-1}"
export VLM_TIMING_SYNC="${VLM_TIMING_SYNC:-1}"
export VLM_STAGE_TIMING="${VLM_STAGE_TIMING:-1}"
export VLM_STAGE_TIMING_DEVICE="${VLM_STAGE_TIMING_DEVICE:-auto}"
export VLM_STAGE_TIMING_SYNC="${VLM_STAGE_TIMING_SYNC:-0}"
export VLM_PRUNE_TIMING="${VLM_PRUNE_TIMING:-1}"
export QWEN3VL_PROFILE_FLOPS="${QWEN3VL_PROFILE_FLOPS:-1}"
export QWEN3VL_RUNTIME_TRACKING="${QWEN3VL_RUNTIME_TRACKING:-1}"
export SEED="${SEED:-42}"
export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-off}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-5}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"

MODEL="${MODEL:-Qwen3-VL-4B-Instruct}"
DATASET_LIST=("${DATASET:-Mind2Web_test_task}")
HISTORY_ABLATION_MODES=(${HISTORY_ABLATION_MODES:-legacy none text thumbnail roi})
export MIND2WEB_ROI_ONLY_HISTORY_ACTION_SOURCE="${MIND2WEB_ROI_ONLY_HISTORY_ACTION_SOURCE:-gt}"
export MIND2WEB_TEXT_HISTORY_ACTION_SOURCE="${MIND2WEB_TEXT_HISTORY_ACTION_SOURCE:-gt}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/qwen3_5/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/qwen3_5/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
TAG="${TAG:-hist4_history_modal_ablation}"
WORK_DIR="${WORK_DIR:-OUTPUT_HIST_ABLATION/mind2web_${TAG}}"
TS="$(date +%Y%m%d_%H%M%S)"
PORT_BASE="${PORT_BASE:-29760}"
REUSE="${REUSE:-0}"

run_one() {
  local dataset_name="$1" mode="$2" port="$3"
  local output_dir="${WORK_DIR}/${mode}/${dataset_name}"
  local log_file="${WORK_DIR}/run_${TS}_${dataset_name}_${mode}.log"
  local action_source="gt"
  export MIND2WEB_HISTORY_ABLATION_MODE="${mode}"
  export MIND2WEB_STATE_PACKET_ENABLE=0
  case "${mode}" in
    legacy) export MIND2WEB_STATE_PACKET_ENABLE=1 ;;
    thumbnail|roi)
      export MIND2WEB_STATE_PACKET_ENABLE=1
      [[ "${mode}" == "roi" ]] && action_source="${MIND2WEB_ROI_ONLY_HISTORY_ACTION_SOURCE,,}"
      ;;
    text) action_source="${MIND2WEB_TEXT_HISTORY_ACTION_SOURCE,,}" ;;
    none) ;;
    *) echo "Unsupported HISTORY_ABLATION_MODES entry: ${mode}" >&2; exit 2 ;;
  esac
  case "${action_source}" in gt|prediction) ;; *) echo "Unsupported Mind2Web history action source: ${action_source}" >&2; exit 2 ;; esac
  export MIND2WEB_HISTORY_ACTION_SOURCE="${action_source}"
  mkdir -p "${output_dir}"
  {
    echo "[HistoryAblation] tag=${TAG} mode=${mode} model=${MODEL} dataset=${dataset_name}"
    echo "[HistoryAblation] output_dir=${output_dir} history_action_source=${MIND2WEB_HISTORY_ACTION_SOURCE}"
    echo "[HistoryAblation] history_steps=${MIND2WEB_HIS_NUM} state_packet=${MIND2WEB_STATE_PACKET_ENABLE}"
    echo "[HistoryAblation] max_new_tokens=${QWEN3VL_MAX_NEW_TOKENS} sample_mode=${VLM_EVAL_SAMPLE_MODE}"
  } | tee -a "${log_file}"
  local reuse_args=()
  [[ "${REUSE}" == "1" ]] && reuse_args+=(--reuse)
  "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" --master_port="${port}" run.py \
    --data "${dataset_name}" --model "${MODEL}" --work-dir "${output_dir}" --mode all \
    "${reuse_args[@]}" 2>&1 | tee -a "${log_file}"
}

port="${PORT_BASE}"
for dataset_name in "${DATASET_LIST[@]}"; do
  for mode in "${HISTORY_ABLATION_MODES[@]}"; do
    run_one "${dataset_name}" "${mode}" "${port}"
    port="$((port + 1))"
  done
done
