#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7

BASE_MODEL="${BASE_MODEL:-/mnt/storage/users/ytshen_data/Qwen3-VL-4B-Instruct}"
ADAPTER_PATH="/home/ytshen/storage_net2/VLMEvalKit_Prune_my/phase1_qwen3vl_with_coord/outputs/grid9_coord_qwen3vl4b_lora_10epoch/epoch_3_adapter" #"${ADAPTER_PATH:-none}"
DATASET="${DATASET:-AndroidControl_Curated_High_Task_Improved}"
SUBSET_LIMIT="${SUBSET_LIMIT:-0}"
DECODER_LAYERS_TO_RUN="${DECODER_LAYERS_TO_RUN:-16}"
TARGET_LAYER_INDEX="${TARGET_LAYER_INDEX:--1}"
TOPK_EVAL="${TOPK_EVAL:-4}"
ATTN_QUERY_CHUNK_SIZE="${ATTN_QUERY_CHUNK_SIZE:-128}"
PROMPT_MODE="${PROMPT_MODE:-grid9_coord}"
ONLY_CLICK_LONGPRESS="${ONLY_CLICK_LONGPRESS:-1}"
SAVE_VISUALIZATIONS=1
VISUALIZE_EVERY=100
VISUALIZE_LIMIT="${VISUALIZE_LIMIT:-0}"
PRINT_PER_SAMPLE=0
PRINT_TIMING=0
LINE_WIDTH="${LINE_WIDTH:-1}"
LINE_COLOR="${LINE_COLOR:-255,64,64}"
OUT_DIR="${OUT_DIR:-OUTPUT/outputs_qwen3vl_android_control_attn_top4_4B_first15_ONLY_CLICK_LONGPRESS_node5_vis/${DATASET}}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_PATH="${LOG_PATH:-run_output_${TIMESTAMP}_android_control_attn_top4_4B_first15_ONLY_CLICK_LONGPRESS_node5_vis.log}"

export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images}"

mkdir -p "$OUT_DIR"

CMD=(
  python -u utils/eval_android_control_attn_top4.py
  --base_model "$BASE_MODEL"
  --dataset "$DATASET"
  --output_dir "$OUT_DIR"
  --subset_limit "$SUBSET_LIMIT"
  --decoder_layers_to_run "$DECODER_LAYERS_TO_RUN"
  --target_layer_index "$TARGET_LAYER_INDEX"
  --topk_eval "$TOPK_EVAL"
  --attn_query_chunk_size "$ATTN_QUERY_CHUNK_SIZE"
  --prompt_mode "$PROMPT_MODE"
  --line_width "$LINE_WIDTH"
  --line_color "$LINE_COLOR"
  --visualize_every "$VISUALIZE_EVERY"
  --visualize_limit "$VISUALIZE_LIMIT"
)

if [[ -n "$ADAPTER_PATH" && "$ADAPTER_PATH" != "none" && "$ADAPTER_PATH" != "null" ]]; then
  CMD+=(--adapter_path "$ADAPTER_PATH")
fi

if [[ "$ONLY_CLICK_LONGPRESS" == "1" || "$ONLY_CLICK_LONGPRESS" == "true" || "$ONLY_CLICK_LONGPRESS" == "True" ]]; then
  CMD+=(--only_click_longpress)
fi

if [[ "$SAVE_VISUALIZATIONS" == "1" || "$SAVE_VISUALIZATIONS" == "true" || "$SAVE_VISUALIZATIONS" == "True" ]]; then
  CMD+=(--save_visualizations)
fi

if [[ "$PRINT_PER_SAMPLE" == "1" || "$PRINT_PER_SAMPLE" == "true" || "$PRINT_PER_SAMPLE" == "True" ]]; then
  CMD+=(--print_per_sample)
fi

if [[ "$PRINT_TIMING" == "1" || "$PRINT_TIMING" == "true" || "$PRINT_TIMING" == "True" ]]; then
  CMD+=(--print_timing)
fi

{
echo "[INFO] cuda_visible_devices   = $CUDA_VISIBLE_DEVICES"
echo "[INFO] base_model             = $BASE_MODEL"
echo "[INFO] adapter_path           = $ADAPTER_PATH"
echo "[INFO] dataset                = $DATASET"
echo "[INFO] subset_limit           = $SUBSET_LIMIT"
echo "[INFO] decoder_layers_to_run  = $DECODER_LAYERS_TO_RUN"
echo "[INFO] target_layer_index     = $TARGET_LAYER_INDEX"
echo "[INFO] topk_eval              = $TOPK_EVAL"
echo "[INFO] attn_query_chunk_size  = $ATTN_QUERY_CHUNK_SIZE"
echo "[INFO] prompt_mode            = $PROMPT_MODE"
echo "[INFO] only_click_longpress   = $ONLY_CLICK_LONGPRESS"
echo "[INFO] save_visualizations    = $SAVE_VISUALIZATIONS"
echo "[INFO] visualize_every        = $VISUALIZE_EVERY"
echo "[INFO] visualize_limit        = $VISUALIZE_LIMIT"
echo "[INFO] print_per_sample       = $PRINT_PER_SAMPLE"
echo "[INFO] print_timing           = $PRINT_TIMING"
echo "[INFO] line_width             = $LINE_WIDTH"
echo "[INFO] line_color             = $LINE_COLOR"
echo "[INFO] output_dir             = $OUT_DIR"
}  | tee -a "$LOG_PATH"

"${CMD[@]}" 2>&1 | tee -a "$LOG_PATH"
# "${CMD[@]}"

echo "[DONE] Results:"
echo "  - $OUT_DIR/per_sample.json"
echo "  - $OUT_DIR/summary.json"
