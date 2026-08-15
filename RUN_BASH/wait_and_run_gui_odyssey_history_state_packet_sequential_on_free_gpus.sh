#!/usr/bin/env bash
set -euo pipefail

# Wait until enough GPUs are free, then run multiple scripts sequentially on
# the same selected GPU set.
#
# Usage:
#   bash RUN_BASH/wait_and_run_gui_odyssey_history_state_packet_sequential_on_free_gpus.sh
#   bash RUN_BASH/wait_and_run_gui_odyssey_history_state_packet_sequential_on_free_gpus.sh \
#     RUN_BASH/run_gui_odyssey_history_state_packet_qwen3_5.sh \
#     RUN_BASH/run_gui_odyssey_history_ablation_qwen3_5.sh
#
# Useful env vars:
#   GPU_CANDIDATES="0 1 2 3 4 5 6 7"
#   REQUIRED_GPU_COUNT=4
#   GPU_CHECK_INTERVAL=60
#   GPU_FREE_MAX_MEM_MB=1024
#   GPU_FREE_MAX_UTIL=10
#   TARGET_LOG_DIR=OUTPUT/auto_gpu_launch_logs
#   SEQ_STOP_ON_ERROR=1
#   RUN_BASH_BIN=bash
#   FORCE_RUN_ON_GPUS=0,1,2,3
#   NPROC_PER_NODE=4

GPU_CANDIDATES="${GPU_CANDIDATES:-0 1 2 3 4 5 6 7}"
REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-4}"
GPU_CHECK_INTERVAL="${GPU_CHECK_INTERVAL:-60}"
GPU_FREE_MAX_MEM_MB="${GPU_FREE_MAX_MEM_MB:-1024}"
GPU_FREE_MAX_UTIL="${GPU_FREE_MAX_UTIL:-10}"
TARGET_LOG_DIR="${TARGET_LOG_DIR:-OUTPUT/auto_gpu_launch_logs}"
STOP_ON_ERROR="${SEQ_STOP_ON_ERROR:-1}"
BASH_BIN="${RUN_BASH_BIN:-bash}"
FORCE_RUN_ON_GPUS="${FORCE_RUN_ON_GPUS:-}"

DEFAULT_SCRIPTS=(
  "RUN_BASH/run_gui_odyssey_history_state_packet_qwen3_5.sh"
)

SCRIPT_LIST=()
if [[ "$#" -gt 0 ]]; then
  SCRIPT_LIST=("$@")
else
  SCRIPT_LIST=("${DEFAULT_SCRIPTS[@]}")
fi

if [[ "${#SCRIPT_LIST[@]}" -lt 1 ]]; then
  echo "[ERROR] No scripts configured."
  exit 1
fi

for script_path in "${SCRIPT_LIST[@]}"; do
  if [[ ! -f "${script_path}" ]]; then
    echo "[ERROR] Script not found: ${script_path}"
    exit 1
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found in PATH."
  exit 1
fi

mkdir -p "${TARGET_LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

normalize_name() {
  local raw="$1"
  raw="$(basename "$raw")"
  raw="${raw%.sh}"
  raw="${raw// /_}"
  echo "$raw"
}

gpu_is_free() {
  local gpu_id="$1"
  local mem_used util_used process_lines

  mem_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -n 1 | tr -d ' ')"
  util_used="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -n 1 | tr -d ' ')"

  if [[ -z "${mem_used}" || -z "${util_used}" ]]; then
    return 1
  fi

  process_lines="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null || true)"
  process_lines="$(echo "${process_lines}" | sed '/^[[:space:]]*$/d')"

  if [[ "${mem_used}" -le "${GPU_FREE_MAX_MEM_MB}" && "${util_used}" -le "${GPU_FREE_MAX_UTIL}" && -z "${process_lines}" ]]; then
    return 0
  fi
  return 1
}

print_gpu_status() {
  for gpu_id in ${GPU_CANDIDATES}; do
    local line
    line="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null || true)"
    if [[ -n "${line}" ]]; then
      echo "[GPUStatus] ${line}"
    else
      echo "[GPUStatus] gpu=${gpu_id} unavailable"
    fi
  done
}

collect_free_gpus() {
  local free_gpus=()
  local gpu_id

  for gpu_id in ${GPU_CANDIDATES}; do
    if gpu_is_free "${gpu_id}"; then
      free_gpus+=("${gpu_id}")
      if [[ "${#free_gpus[@]}" -ge "${REQUIRED_GPU_COUNT}" ]]; then
        printf '%s\n' "${free_gpus[@]}"
        return 0
      fi
    fi
  done
  return 1
}

run_scripts_on_gpus() {
  local visible_gpus="$1"
  local nproc_per_node="$2"
  local total index failures script_path script_name log_file status

  total="${#SCRIPT_LIST[@]}"
  index=0
  failures=0

  echo "[Launcher] selected CUDA_VISIBLE_DEVICES=${visible_gpus}"
  echo "[Launcher] selected NPROC_PER_NODE=${nproc_per_node}"
  echo "[Launcher] configured scripts:"
  for script_path in "${SCRIPT_LIST[@]}"; do
    echo "  - ${script_path}"
  done

  for script_path in "${SCRIPT_LIST[@]}"; do
    index=$((index + 1))
    script_name="$(normalize_name "${script_path}")"
    log_file="${TARGET_LOG_DIR}/${TIMESTAMP}_gpu${visible_gpus//,/}_$(printf "%02d" "${index}")_${script_name}.log"

    echo "[Launcher] (${index}/${total}) start ${script_path}"
    echo "[Launcher] log -> ${log_file}"

    set +e
    CUDA_VISIBLE_DEVICES="${visible_gpus}" \
    NPROC_PER_NODE="${nproc_per_node}" \
    "${BASH_BIN}" "${script_path}" 2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    set -e

    if [[ "${status}" -eq 0 ]]; then
      echo "[Launcher] (${index}/${total}) done ${script_path}"
    elif [[ "${status}" -eq 130 ]]; then
      echo "[Launcher] (${index}/${total}) interrupted by user status=130 script=${script_path}"
      exit 130
    else
      failures=$((failures + 1))
      echo "[Launcher] (${index}/${total}) failed status=${status} script=${script_path}"
      if [[ "${STOP_ON_ERROR}" == "1" || "${STOP_ON_ERROR}" == "true" || "${STOP_ON_ERROR}" == "True" ]]; then
        echo "[Launcher] stop_on_error=1, aborting."
        exit "${status}"
      fi
    fi
  done

  if [[ "${failures}" -gt 0 ]]; then
    echo "[Launcher] finished with failures=${failures}"
    exit 1
  fi

  echo "[Launcher] all scripts finished successfully."
}

if [[ -n "${FORCE_RUN_ON_GPUS}" ]]; then
  FORCED_NPROC="${NPROC_PER_NODE:-$(echo "${FORCE_RUN_ON_GPUS}" | tr ',' '\n' | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')}"
  echo "[Launcher] FORCE_RUN_ON_GPUS=${FORCE_RUN_ON_GPUS}, skip waiting."
  run_scripts_on_gpus "${FORCE_RUN_ON_GPUS}" "${FORCED_NPROC}"
  exit $?
fi

echo "[Launcher] gpu_candidates=${GPU_CANDIDATES}"
echo "[Launcher] required_gpu_count=${REQUIRED_GPU_COUNT}"
echo "[Launcher] check_interval_s=${GPU_CHECK_INTERVAL}"
echo "[Launcher] free_if_mem_le_mb=${GPU_FREE_MAX_MEM_MB}"
echo "[Launcher] free_if_util_le_pct=${GPU_FREE_MAX_UTIL}"

while true; do
  print_gpu_status
  mapfile -t FREE_GPU_LIST < <(collect_free_gpus || true)
  if [[ "${#FREE_GPU_LIST[@]}" -ge "${REQUIRED_GPU_COUNT}" ]]; then
    SELECTED_GPUS="$(IFS=,; echo "${FREE_GPU_LIST[*]:0:${REQUIRED_GPU_COUNT}}")"
    SELECTED_NPROC="${NPROC_PER_NODE:-${REQUIRED_GPU_COUNT}}"
    echo "[Launcher] detected ${REQUIRED_GPU_COUNT} free GPUs: ${SELECTED_GPUS}"
    run_scripts_on_gpus "${SELECTED_GPUS}" "${SELECTED_NPROC}"
    exit $?
  fi
  echo "[Launcher] no sufficient free GPUs yet, sleep ${GPU_CHECK_INTERVAL}s"
  sleep "${GPU_CHECK_INTERVAL}"
done
