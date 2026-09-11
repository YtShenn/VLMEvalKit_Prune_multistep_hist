#!/usr/bin/env bash
set -euo pipefail

# Three layer prunes (3, 6, 15): 0.669433^3 ~= 0.30 total visual retention.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATASETS="${DATASETS:-GUIOdyssey_high_task_split}"
export SPARSEVLM_RETAIN_RATIO="${SPARSEVLM_RETAIN_RATIO:-0.669433}"
export WORK_DIR="${WORK_DIR:-OUTPUT_SPARSEVLM/4B_gui_odyssey_total_keep0.30_layer_keep${SPARSEVLM_RETAIN_RATIO}}"
exec bash "${ROOT_DIR}/RUN_BASH/run_qwen3vl_sparsevlm_4datasets_4B.sh"
