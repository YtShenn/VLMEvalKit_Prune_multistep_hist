#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper around the complete, timed four-dataset runner.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATASETS="${DATASETS:-AndroidControl_Curated_High_Task_Improved}"
export WORK_DIR="${WORK_DIR:-OUTPUT_HISTPRUNE/4B_android_control}"
exec bash "${ROOT_DIR}/RUN_BASH/run_qwen3vl_histprune_4datasets_4B.sh"
