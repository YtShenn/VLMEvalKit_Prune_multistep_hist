#!/usr/bin/env bash
# Deployment-faithful redundancy--utility experiment: remove exact post-merge
# visual tokens for each sampled local region or full history frame. The
# original image pixels remain untouched; pruning is applied inside Qwen's
# KV/M-RoPE/DeepStack-safe decoder sequence-surgery path.
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export INTERVENTION_MODE=token_prune
export REGION_SCALES="${REGION_SCALES:-4,global}"
export TOKEN_PRUNE_LAYER="${TOKEN_PRUNE_LAYER:-0}"
DATASET="${DATASET:-AndroidControl_Curated_High_Task_Improved}"
WORK_DIR="${WORK_DIR:-OUTPUT_history_redundancy_utility_realdrop/${DATASET}}"

exec bash "$SCRIPT_DIR/run_history_redundancy_utility.sh"
