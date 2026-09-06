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

export QWEN3VL_ENABLE_TEMPLATE_PREFILL="${QWEN3VL_ENABLE_TEMPLATE_PREFILL:-0}"
export QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE="1"
export QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG="${QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG:-1}"
export QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG_POS="${QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG_POS:-0}"
export QWEN3VL_TEMPLATE_PREFILL_IMPL="${QWEN3VL_TEMPLATE_PREFILL_IMPL:-stateful}" #single_generate
export QWEN3VL_TEMPLATE_PREFILL_DATASETS="${QWEN3VL_TEMPLATE_PREFILL_DATASETS:-androidcontrol}"
export QWEN3VL_TEMPLATE_PREFILL_DEBUG="${QWEN3VL_TEMPLATE_PREFILL_DEBUG:-1}"
export QWEN3VL_TEMPLATE_PREFILL_ANDROID_MODE="${QWEN3VL_TEMPLATE_PREFILL_ANDROID_MODE:-action_first_json}" #bbox_first_json

export QWEN3VL_ENABLE_ROI_PRUNE=0
export QWEN3VL_ROI_PRUNE_USE_CACHE="${QWEN3VL_ROI_PRUNE_USE_CACHE:-1}"
export QWEN3VL_ROI_PRUNE_PRINT_PER_SAMPLE="${QWEN3VL_ROI_PRUNE_PRINT_PER_SAMPLE:-1}"

export QWEN3VL_ENABLE_ATTN_PRUNE="${QWEN3VL_ENABLE_ATTN_PRUNE:-0}"
export QWEN3VL_ATTN_PRUNE_LAYERS="${QWEN3VL_ATTN_PRUNE_LAYERS:-1}"
export QWEN3VL_ATTN_PRUNE_VIS_LAYERS="${QWEN3VL_ATTN_PRUNE_VIS_LAYERS:-1,2,3,5,7,15,23}"
export QWEN3VL_ATTN_PRUNE_KEEP_RATIO="${QWEN3VL_ATTN_PRUNE_KEEP_RATIO:-0.6}"
export QWEN3VL_ATTN_PRUNE_USE_CACHE="${QWEN3VL_ATTN_PRUNE_USE_CACHE:-1}"
export QWEN3VL_ATTN_PRUNE_VIS="${QWEN3VL_ATTN_PRUNE_VIS:-0}"        #attn可视化开关
export QWEN3VL_ATTN_PRUNE_STATS="${QWEN3VL_ATTN_PRUNE_STATS:-0}"    #分布尖锐程度统计
export QWEN3VL_ATTN_PRUNE_OUT_DIR="${QWEN3VL_ATTN_PRUNE_OUT_DIR:-OUTPUT/attn_prune_debug/android_control_history_state_packet_inst_after_siedeattn_high_prune_vispred}"
export QWEN3VL_ATTN_PRUNE_PRED_DETAIL_JSON="${QWEN3VL_ATTN_PRUNE_PRED_DETAIL_JSON:-OUTPUT_0/ablation_android_control_hist4_keep_system_prompt_state_packet_official_xiabanqian_node5_0812(only_HTI)_ratio2/AndroidControl_Curated_High_Task_Improved/Qwen3-VL-4B-Instruct/T20260812_G1e13089b/Qwen3-VL-4B-Instruct_AndroidControl_Curated_High_Task_Improved_android_control_detail_official.json}"
export QWEN3VL_ATTN_PRUNE_DEBUG="${QWEN3VL_ATTN_PRUNE_DEBUG:-1}"
export QWEN3VL_ATTN_PRUNE_STRICT_QUERY_MARKER="${QWEN3VL_ATTN_PRUNE_STRICT_QUERY_MARKER:-1}"
export QWEN3VL_ATTN_PRUNE_PRINT_QUERY_TOKENS="${QWEN3VL_ATTN_PRUNE_PRINT_QUERY_TOKENS:-0}"
export QWEN3VL_ATTN_PRUNE_SIDE_COMPUTE="${QWEN3VL_ATTN_PRUNE_SIDE_COMPUTE:-1}" #置1是额外再算attn，置0是直接抓取，用eager attn，会慢一些
export QWEN3VL_ATTN_PRUNE_PRUNE_VIS="${QWEN3VL_ATTN_PRUNE_PRUNE_VIS:-0}" #剪枝后可视化，保留token显示attn，剪掉token标黑
export QWEN3VL_ATTN_PRUNE_SAFETY_KEEP="${QWEN3VL_ATTN_PRUNE_SAFETY_KEEP:-0}" #额外保留均匀/边缘/中心/文字密集token
export QWEN3VL_ATTN_PRUNE_SAFETY_MAX_EXTRA_RATIO="${QWEN3VL_ATTN_PRUNE_SAFETY_MAX_EXTRA_RATIO:-0.20}"
export QWEN3VL_ATTN_PRUNE_SAFETY_MAX_KEEP_RATIO="${QWEN3VL_ATTN_PRUNE_SAFETY_MAX_KEEP_RATIO:-1.0}"
export QWEN3VL_ATTN_PRUNE_SAFETY_SELECT_MODE="${QWEN3VL_ATTN_PRUNE_SAFETY_SELECT_MODE:-priority}" #priority/spatial/attn
export QWEN3VL_ATTN_PRUNE_SAFETY_ATTN_TIE_WEIGHT="${QWEN3VL_ATTN_PRUNE_SAFETY_ATTN_TIE_WEIGHT:-0.25}"
export QWEN3VL_ATTN_PRUNE_SAFETY_UNIFORM_RATIO="${QWEN3VL_ATTN_PRUNE_SAFETY_UNIFORM_RATIO:-0.05}"
export QWEN3VL_ATTN_PRUNE_SAFETY_TOP_RATIO="${QWEN3VL_ATTN_PRUNE_SAFETY_TOP_RATIO:-0.08}"
export QWEN3VL_ATTN_PRUNE_SAFETY_BOTTOM_RATIO="${QWEN3VL_ATTN_PRUNE_SAFETY_BOTTOM_RATIO:-0.14}"
export QWEN3VL_ATTN_PRUNE_SAFETY_SIDE_RATIO="${QWEN3VL_ATTN_PRUNE_SAFETY_SIDE_RATIO:-0.04}"
export QWEN3VL_ATTN_PRUNE_SAFETY_CENTER_H_RATIO="${QWEN3VL_ATTN_PRUNE_SAFETY_CENTER_H_RATIO:-0.22}"
export QWEN3VL_ATTN_PRUNE_SAFETY_CENTER_W_RATIO="${QWEN3VL_ATTN_PRUNE_SAFETY_CENTER_W_RATIO:-0.28}"
export QWEN3VL_ATTN_PRUNE_SAFETY_TEXT_DENSE_RATIO="${QWEN3VL_ATTN_PRUNE_SAFETY_TEXT_DENSE_RATIO:-0.05}"

export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-task}"
export VLM_EVAL_SAMPLE_INDEX_FILE="${VLM_EVAL_SAMPLE_INDEX_FILE:-OUTPUT_0/attn_prune_debug/android_control_history_state_packet_step_instruct/selected_indices.txt}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-10}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"

export ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS="${ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS:-1}"
export ANDROID_CONTROL_MAX_HISTORY_IMAGES="${ANDROID_CONTROL_MAX_HISTORY_IMAGES:-4}"
export ANDROID_CONTROL_HISTORY_KEEP_SYSTEM_PROMPT="${ANDROID_CONTROL_HISTORY_KEEP_SYSTEM_PROMPT:-0}"

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

MODEL="${MODEL:-Qwen3-VL-4B-Instruct-AttnPrune}"
DATASET_LIST=(
    # "AndroidControl_Curated_High_Point"
    "AndroidControl_Curated_High_Task_Improved"
    # "AndroidControl_Curated_Low_BBox"
)
PYTHON_BIN="${PYTHON_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/torchrun}"
TS="$(date +%Y%m%d_%H%M%S)"
# TAG="${TAG:-hist4_keep_system_prompt_state_packet_attn_prune_structured_fast_official_nonvis_prune_layer${QWEN3VL_ATTN_PRUNE_LAYERS}_${QWEN3VL_ATTN_PRUNE_KEEP_RATIO}_eagerattn_0826_high}"
TAG="${TAG:-hist4_keep_system_prompt_state_packet_fast_decode_official_nonvis_high_0906}"
# TAG="${TAG:-hist4_keep_system_prompt_baseline_official_nonvis_Low_BBox_0825}"
# TAG="${TAG:-hist4_keep_system_prompt_baseline_official_nonvis_high_0828}"
# TAG="${TAG:-hist4_keep_system_prompt_state_packet_attn_prune_structured_fast_official_prune_layer${QWEN3VL_ATTN_PRUNE_LAYERS}_${QWEN3VL_ATTN_PRUNE_KEEP_RATIO}_sideattn_0830_high}"
# TAG="${TAG:-hist4_keep_system_prompt_baseline_official_nonhistory_atall_high_0906_task}"

WORK_DIR="${WORK_DIR:-OUTPUT/android_control_${TAG}}"
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
    echo "[Run] structured_fast_decode=${QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE}"
    echo "[Run] structured_fast_decode_debug=${QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG}"
    echo "[Run] structured_fast_decode_debug_pos=${QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG_POS}"
    echo "[Run] template_prefill_impl=${QWEN3VL_TEMPLATE_PREFILL_IMPL}"
    echo "[Run] template_prefill_datasets=${QWEN3VL_TEMPLATE_PREFILL_DATASETS}"
    echo "[Run] template_prefill_android_mode=${QWEN3VL_TEMPLATE_PREFILL_ANDROID_MODE}"
    echo "[Run] attn_prune_enable=${QWEN3VL_ENABLE_ATTN_PRUNE}"
    echo "[Run] attn_prune_layers=${QWEN3VL_ATTN_PRUNE_LAYERS}"
    echo "[Run] attn_prune_vis_layers=${QWEN3VL_ATTN_PRUNE_VIS_LAYERS}"
    echo "[Run] attn_prune_keep_ratio=${QWEN3VL_ATTN_PRUNE_KEEP_RATIO}"
    echo "[Run] attn_prune_use_cache=${QWEN3VL_ATTN_PRUNE_USE_CACHE}"
    echo "[Run] attn_prune_vis=${QWEN3VL_ATTN_PRUNE_VIS}"
    echo "[Run] attn_prune_stats=${QWEN3VL_ATTN_PRUNE_STATS}"
    echo "[Run] attn_prune_out_dir=${QWEN3VL_ATTN_PRUNE_OUT_DIR}"
    echo "[Run] attn_prune_pred_detail_json=${QWEN3VL_ATTN_PRUNE_PRED_DETAIL_JSON}"
    echo "[Run] attn_prune_strict_query_marker=${QWEN3VL_ATTN_PRUNE_STRICT_QUERY_MARKER}"
    echo "[Run] attn_prune_print_query_tokens=${QWEN3VL_ATTN_PRUNE_PRINT_QUERY_TOKENS}"
    echo "[Run] attn_prune_side_compute=${QWEN3VL_ATTN_PRUNE_SIDE_COMPUTE}"
    echo "[Run] attn_prune_prune_vis=${QWEN3VL_ATTN_PRUNE_PRUNE_VIS}"
    echo "[Run] attn_prune_safety_keep=${QWEN3VL_ATTN_PRUNE_SAFETY_KEEP}"
    echo "[Run] attn_prune_safety_max_extra_ratio=${QWEN3VL_ATTN_PRUNE_SAFETY_MAX_EXTRA_RATIO}"
    echo "[Run] attn_prune_safety_max_keep_ratio=${QWEN3VL_ATTN_PRUNE_SAFETY_MAX_KEEP_RATIO}"
    echo "[Run] attn_prune_safety_select_mode=${QWEN3VL_ATTN_PRUNE_SAFETY_SELECT_MODE}"
    echo "[Run] attn_prune_safety_attn_tie_weight=${QWEN3VL_ATTN_PRUNE_SAFETY_ATTN_TIE_WEIGHT}"
    echo "[Run] attn_prune_safety_uniform_ratio=${QWEN3VL_ATTN_PRUNE_SAFETY_UNIFORM_RATIO}"
    echo "[Run] attn_prune_safety_top_ratio=${QWEN3VL_ATTN_PRUNE_SAFETY_TOP_RATIO}"
    echo "[Run] attn_prune_safety_bottom_ratio=${QWEN3VL_ATTN_PRUNE_SAFETY_BOTTOM_RATIO}"
    echo "[Run] attn_prune_safety_side_ratio=${QWEN3VL_ATTN_PRUNE_SAFETY_SIDE_RATIO}"
    echo "[Run] attn_prune_safety_center_h_ratio=${QWEN3VL_ATTN_PRUNE_SAFETY_CENTER_H_RATIO}"
    echo "[Run] attn_prune_safety_center_w_ratio=${QWEN3VL_ATTN_PRUNE_SAFETY_CENTER_W_RATIO}"
    echo "[Run] attn_prune_safety_text_dense_ratio=${QWEN3VL_ATTN_PRUNE_SAFETY_TEXT_DENSE_RATIO}"
    echo "[Run] sample_mode=${VLM_EVAL_SAMPLE_MODE}"
    echo "[Run] sample_index_file=${VLM_EVAL_SAMPLE_INDEX_FILE:-}"
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
