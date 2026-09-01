#!/usr/bin/env bash
set -euo pipefail

# Run the same AndroidControl history/state-packet setting as the baseline
# confidence observer, but enable current-frame attention pruning at 50% keep.
export QWEN3VL_ENABLE_ATTN_PRUNE="${QWEN3VL_ENABLE_ATTN_PRUNE:-1}"
export QWEN3VL_ATTN_PRUNE_LAYERS="${QWEN3VL_ATTN_PRUNE_LAYERS:-3}"
export QWEN3VL_ATTN_PRUNE_KEEP_RATIO="${QWEN3VL_ATTN_PRUNE_KEEP_RATIO:-0.5}"
export QWEN3VL_ATTN_CONF_ENABLE="${QWEN3VL_ATTN_CONF_ENABLE:-1}"
export QWEN3VL_ATTN_CONF_LAYERS="${QWEN3VL_ATTN_CONF_LAYERS:-0,1,2,3}"
export QWEN3VL_ATTN_CONF_ANALYZE="${QWEN3VL_ATTN_CONF_ANALYZE:-1}"

export TAG="${TAG:-hist4_keep_system_prompt_state_packet_attn_prune50_conf_observe_layers${QWEN3VL_ATTN_CONF_LAYERS//,/}_layer${QWEN3VL_ATTN_PRUNE_LAYERS}_high_all}"
export WORK_DIR="${WORK_DIR:-OUTPUT/android_control_${TAG}}"

exec RUN_BASH/run_android_control_history_state_packet_qwen3_5_attn_confidence_analysis.sh
