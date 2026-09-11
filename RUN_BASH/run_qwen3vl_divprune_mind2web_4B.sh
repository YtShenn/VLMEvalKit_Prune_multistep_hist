#!/usr/bin/env bash
set -euo pipefail

# DivPrune runs once before decoder layer 0, so this is total visual retention.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATASETS="${DATASETS:-Mind2Web_test_task}"
export DIVPRUNE_KEEP_RATIO="${DIVPRUNE_KEEP_RATIO:-0.40}"
export WORK_DIR="${WORK_DIR:-OUTPUT_DIVPRUNE/4B_mind2web_total_keep${DIVPRUNE_KEEP_RATIO}}"
exec bash "${ROOT_DIR}/RUN_BASH/run_qwen3vl_divprune_4B.sh"
