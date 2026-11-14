#!/usr/bin/env bash
# Controlled crop-policy ablation: History Packet only, normal decoding.
# Existing HP/SFD launchers remain unchanged.
set -euo pipefail

export ANDROID_CONTROL_STATE_PACKET_CROP_POLICY="${ANDROID_CONTROL_STATE_PACKET_CROP_POLICY:-action}"
export QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE=0
export QWEN3VL_ENABLE_TEMPLATE_PREFILL=0
export QWEN3VL_ENABLE_ROI_PRUNE=0
export QWEN3VL_ENABLE_ATTN_PRUNE=0

if [[ -z "${TAG:-}" ]]; then
  export TAG="hist4_state_packet_crop_${ANDROID_CONTROL_STATE_PACKET_CROP_POLICY}_normal_decode"
fi

echo "[CropPolicyAblation] policy=${ANDROID_CONTROL_STATE_PACKET_CROP_POLICY} SFD=${QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE} template_prefill=${QWEN3VL_ENABLE_TEMPLATE_PREFILL} roi_prune=${QWEN3VL_ENABLE_ROI_PRUNE} attn_prune=${QWEN3VL_ENABLE_ATTN_PRUNE}"
exec bash "$(dirname "$0")/run_android_control_history_state_packet_only_hist4.sh"
