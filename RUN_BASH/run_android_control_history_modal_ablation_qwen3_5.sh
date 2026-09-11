#!/usr/bin/env bash
# Run AndroidControl history-retention ablations:
# legacy (thumbnail + ROI) / none / text / thumbnail / roi.
# Edit the configuration block below directly, then run this script without
# command-line parameters.
set -euo pipefail

# ============================ Edit this block ============================
export CUDA_VISIBLE_DEVICES=6
export ANDROID_CONTROL_CURATED_EVAL_MODE=official
export ANDROID_CONTROL_CURATED_ROOT="/mnt/storage2/users/ytshen_data/AndroidControl_Curated"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images"
export ANDROID_CONTROL_SEQUENTIAL_ORDER=1

# Print the complete prompt for every sample.  The output includes
# history_ablation_mode, history_images, history_text, and prompt_begin/end.
export ANDROID_CONTROL_DEBUG_HISTORY_PROMPT=1
# Optional extra state-packet metadata (ROI box and visual-token estimates).
export ANDROID_CONTROL_STATE_PACKET_DEBUG=0

export QWEN3VL_ANDROID_DENORM_ON_INFER=1
export QWEN3VL_ANDROID_DENORM_BASE=1000
export QWEN3VL_ENABLE_ROI_PRUNE=0

# Use a fixed four-step history protocol for every condition.  The `none` and
# `text` modes override visual history internally, while thumbnail/roi retain
# exactly this many historical steps.
export ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS=1
export ANDROID_CONTROL_MAX_HISTORY_IMAGES=4
# Keep the pre-existing multimodal prompt protocol.  With prompt debugging
# enabled, this prints every history packet's actual image path/type/ROI box.
export ANDROID_CONTROL_HISTORY_KEEP_SYSTEM_PROMPT=1
# `legacy` is the complete state-packet baseline: thumbnail + action ROI for
# every history step.  thumbnail/roi override its image selection internally;
# none/text suppress history visuals internally.
export ANDROID_CONTROL_STATE_PACKET_ENABLE=1
export ANDROID_CONTROL_STATE_PACKET_CACHE_DIR="/tmp/android_control_state_packet_cache"
export ANDROID_CONTROL_STATE_PACKET_PATCH_SIZE=16
export ANDROID_CONTROL_STATE_PACKET_MERGE_SIZE=2
export ANDROID_CONTROL_STATE_PACKET_THUMB_LONG_EDGE=192
export ANDROID_CONTROL_STATE_PACKET_ROI_LONG_EDGE=224
export ANDROID_CONTROL_STATE_PACKET_ROI_SHORT_SIDE_RATIO=0.22
export ANDROID_CONTROL_STATE_PACKET_ROI_MIN_SIDE_PX=160

export VLM_TIMING=1
export VLM_TIMING_VERBOSE=1
export VLM_TIMING_SYNC=1
export VLM_STAGE_TIMING=1
export VLM_STAGE_TIMING_DEVICE=auto
export VLM_STAGE_TIMING_SYNC=0
export VLM_PRUNE_TIMING=1
export SEED=42

# Set VLM_EVAL_SAMPLE_MODE=all for a complete evaluation.
export VLM_EVAL_SAMPLE_MODE=task
export VLM_EVAL_SAMPLE_TASKS=5
export VLM_EVAL_SAMPLE_SEED=42

MODEL="Qwen3-VL-4B-Instruct"
DATASET_LIST=(
  "AndroidControl_Curated_High_Task_Improved"
)
# HISTORY_ABLATION_MODES=(none text)
# HISTORY_ABLATION_MODES=(thumbnail roi legacy)
HISTORY_ABLATION_MODES=(legacy)
# 默认 legacy
# export ANDROID_CONTROL_HISTORY_ABLATION_MODE=none       # 无历史文本、无历史图像
# export ANDROID_CONTROL_HISTORY_ABLATION_MODE=text       # 仅历史动作文本
# export ANDROID_CONTROL_HISTORY_ABLATION_MODE=thumbnail  # 历史动作文本 + 每步仅缩略图
# export ANDROID_CONTROL_HISTORY_ABLATION_MODE=roi        # 历史动作文本 + 每步仅 ROI
PYTHON_BIN="/home/ytshen/anaconda3/envs/qwen3_5/bin/python"
TORCHRUN_BIN="/home/ytshen/anaconda3/envs/qwen3_5/bin/torchrun"
TAG="hist4_history_modal_ablation"
WORK_DIR="OUTPUT/ablation_android_control_${TAG}"
TS="$(date +%Y%m%d_%H%M%S)"
PORT_BASE=29720
REUSE=0
# ========================================================================

run_one() {
  local dataset_name="$1"
  local mode="$2"
  local port="$3"
  local output_dir="${WORK_DIR}/${mode}/${dataset_name}"
  local log_file="${WORK_DIR}/run_${TS}_${dataset_name}_${mode}.log"

  export ANDROID_CONTROL_HISTORY_ABLATION_MODE="${mode}"
  mkdir -p "${output_dir}"
  {
    echo "[HistoryAblation] tag=${TAG}"
    echo "[HistoryAblation] mode=${ANDROID_CONTROL_HISTORY_ABLATION_MODE}"
    echo "[HistoryAblation] model=${MODEL} dataset=${dataset_name}"
    echo "[HistoryAblation] output_dir=${output_dir}"
    echo "[HistoryAblation] max_history=${ANDROID_CONTROL_MAX_HISTORY_IMAGES}"
    echo "[HistoryAblation] debug_history_prompt=${ANDROID_CONTROL_DEBUG_HISTORY_PROMPT}"
    echo "[HistoryAblation] state_packet_debug=${ANDROID_CONTROL_STATE_PACKET_DEBUG}"
    echo "[HistoryAblation] state_packet_enable=${ANDROID_CONTROL_STATE_PACKET_ENABLE}"
    echo "[HistoryAblation] thumbnail_edge=${ANDROID_CONTROL_STATE_PACKET_THUMB_LONG_EDGE}"
    echo "[HistoryAblation] roi_edge=${ANDROID_CONTROL_STATE_PACKET_ROI_LONG_EDGE}"
    echo "[HistoryAblation] sample_mode=${VLM_EVAL_SAMPLE_MODE} sample_tasks=${VLM_EVAL_SAMPLE_TASKS}"
  } | tee -a "${log_file}"

  local reuse_args=()
  if [[ "${REUSE}" == "1" ]]; then
    reuse_args+=(--reuse)
  fi

  "${TORCHRUN_BIN}" --standalone --nproc_per_node=1 --master_port="${port}" run.py \
    --data "${dataset_name}" \
    --model "${MODEL}" \
    --work-dir "${output_dir}" \
    --mode all \
    "${reuse_args[@]}" 2>&1 | tee -a "${log_file}"
}

port="${PORT_BASE}"
for dataset_name in "${DATASET_LIST[@]}"; do
  for mode in "${HISTORY_ABLATION_MODES[@]}"; do
    case "${mode}" in
      legacy|none|text|thumbnail|roi) ;;
      *)
        echo "Unsupported HISTORY_ABLATION_MODES entry: ${mode}" >&2
        exit 2
        ;;
    esac
    run_one "${dataset_name}" "${mode}" "${port}"
    port="$((port + 1))"
  done
done
