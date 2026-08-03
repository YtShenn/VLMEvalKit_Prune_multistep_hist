#!/usr/bin/env bash
set -euo pipefail

# Run multiple bash scripts sequentially.
# You can either:
# 1. Edit DEFAULT_SCRIPTS below, then run:
#      bash RUN_BASH/run_scripts_sequential.sh
# 2. Or override from command line:
#      bash RUN_BASH/run_scripts_sequential.sh script1.sh script2.sh
#
# Optional env vars:
#   SEQ_STOP_ON_ERROR=1   stop immediately on first failure (default)
#   SEQ_LOG_DIR=...       directory to save per-script logs
#   SEQ_BASH_BIN=bash     shell executable used to launch each script

DEFAULT_SCRIPTS=(
    "RUN_BASH/run_qwen3vl_gui_odyssey_4B_json_top4_uniform_prune.sh"
    "RUN_BASH/run_qwen3vl_gui_odyssey_4B_json_top4_uniform_prune.sh"
#   "RUN_BASH/run_qwen3vl_gui_odyssey_4B_json_top4_uniform_prune.sh"
#   "RUN_BASH/run_qwen3vl_gui_odyssey_4B_template_prefill_timing.sh"
#   "RUN_BASH/run_qwen3vl_gui_odyssey_4B_baseline_timing_sampled.sh"
)

STOP_ON_ERROR="${SEQ_STOP_ON_ERROR:-1}"
LOG_DIR="${SEQ_LOG_DIR:-OUTPUT/sequential_run_logs}"
BASH_BIN="${SEQ_BASH_BIN:-bash}"

on_interrupt() {
  echo "[Sequential] interrupted by user (SIGINT)."
  exit 130
}

trap on_interrupt INT

SCRIPT_LIST=()
if [[ "$#" -gt 0 ]]; then
  SCRIPT_LIST=("$@")
else
  SCRIPT_LIST=("${DEFAULT_SCRIPTS[@]}")
fi

if [[ "${#SCRIPT_LIST[@]}" -lt 1 ]]; then
  echo "[ERROR] No scripts configured."
  echo "Please either:"
  echo "  1. Edit DEFAULT_SCRIPTS in $0"
  echo "  2. Or pass scripts from command line"
  exit 1
fi

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

normalize_name() {
  local raw="$1"
  raw="$(basename "$raw")"
  raw="${raw%.sh}"
  raw="${raw// /_}"
  echo "$raw"
}

failures=0
index=0
total="$#"

total="${#SCRIPT_LIST[@]}"

echo "[Sequential] configured scripts:"
for script_path in "${SCRIPT_LIST[@]}"; do
  echo "  - ${script_path}"
done

for script_path in "${SCRIPT_LIST[@]}"; do
  index=$((index + 1))

  if [[ ! -f "${script_path}" ]]; then
    echo "[ERROR] Script not found: ${script_path}"
    exit 1
  fi

  script_name="$(normalize_name "${script_path}")"
  log_file="${LOG_DIR}/${TIMESTAMP}_$(printf "%02d" "${index}")_${script_name}.log"

  echo "[Sequential] (${index}/${total}) start ${script_path}"
  echo "[Sequential] log -> ${log_file}"

  set +e
  "${BASH_BIN}" "${script_path}" 2>&1 | tee "${log_file}"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -eq 0 ]]; then
    echo "[Sequential] (${index}/${total}) done ${script_path}"
  elif [[ "${status}" -eq 130 ]]; then
    echo "[Sequential] (${index}/${total}) interrupted by user status=130 script=${script_path}"
    exit 130
  else
    failures=$((failures + 1))
    echo "[Sequential] (${index}/${total}) failed status=${status} script=${script_path}"
    if [[ "${STOP_ON_ERROR}" == "1" || "${STOP_ON_ERROR}" == "true" || "${STOP_ON_ERROR}" == "True" ]]; then
      echo "[Sequential] stop_on_error=1, aborting."
      exit "${status}"
    fi
  fi
done

if [[ "${failures}" -gt 0 ]]; then
  echo "[Sequential] finished with failures=${failures}"
  exit 1
fi

echo "[Sequential] all scripts finished successfully."
