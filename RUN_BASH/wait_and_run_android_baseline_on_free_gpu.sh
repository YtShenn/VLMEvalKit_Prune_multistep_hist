#!/usr/bin/env bash
set -euo pipefail

# Poll GPUs 4/5/6/7 and launch the Android baseline timing script
# as soon as one GPU is considered free.
#
# Default free criteria:
# - memory.used <= 1024 MiB
# - utilization.gpu <= 10%
# - no running compute processes reported by nvidia-smi
#
# Useful env vars:
#   GPU_CANDIDATES="4 5 6 7"
#   GPU_CHECK_INTERVAL=60
#   GPU_FREE_MAX_MEM_MB=1024
#   GPU_FREE_MAX_UTIL=10
#   TARGET_SCRIPT=RUN_BASH/run_qwen3vl_android_control_curated_4B_baseline_timing_sampled.sh
#   TARGET_LOG_DIR=OUTPUT/auto_gpu_launch_logs
#   FORCE_RUN_ON_GPU=6         # bypass waiting and run immediately on this GPU

GPU_CANDIDATES="${GPU_CANDIDATES:-4 5 6 7}"
GPU_CHECK_INTERVAL="${GPU_CHECK_INTERVAL:-60}"
GPU_FREE_MAX_MEM_MB="${GPU_FREE_MAX_MEM_MB:-1024}"
GPU_FREE_MAX_UTIL="${GPU_FREE_MAX_UTIL:-10}"
TARGET_SCRIPT="${TARGET_SCRIPT:-RUN_BASH/run_qwen3vl_android_control_curated_4B_baseline_timing_sampled.sh}"
TARGET_LOG_DIR="${TARGET_LOG_DIR:-OUTPUT/auto_gpu_launch_logs}"
FORCE_RUN_ON_GPU="${FORCE_RUN_ON_GPU:-}"

mkdir -p "${TARGET_LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "[ERROR] Target script not found: ${TARGET_SCRIPT}"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found in PATH."
  exit 1
fi

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

launch_on_gpu() {
  local gpu_id="$1"
  local script_name log_file
  script_name="$(basename "${TARGET_SCRIPT}")"
  script_name="${script_name%.sh}"
  log_file="${TARGET_LOG_DIR}/${TIMESTAMP}_${script_name}_gpu${gpu_id}.log"

  echo "[Launcher] starting ${TARGET_SCRIPT} on GPU ${gpu_id}"
  echo "[Launcher] log -> ${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" bash "${TARGET_SCRIPT}" 2>&1 | tee "${log_file}"
}

if [[ -n "${FORCE_RUN_ON_GPU}" ]]; then
  echo "[Launcher] FORCE_RUN_ON_GPU=${FORCE_RUN_ON_GPU}, skip waiting."
  launch_on_gpu "${FORCE_RUN_ON_GPU}"
  exit $?
fi

echo "[Launcher] target_script=${TARGET_SCRIPT}"
echo "[Launcher] gpu_candidates=${GPU_CANDIDATES}"
echo "[Launcher] check_interval_s=${GPU_CHECK_INTERVAL}"
echo "[Launcher] free_if_mem_le_mb=${GPU_FREE_MAX_MEM_MB}"
echo "[Launcher] free_if_util_le_pct=${GPU_FREE_MAX_UTIL}"

while true; do
  print_gpu_status
  for gpu_id in ${GPU_CANDIDATES}; do
    if gpu_is_free "${gpu_id}"; then
      echo "[Launcher] detected free GPU ${gpu_id}"
      launch_on_gpu "${gpu_id}"
      exit $?
    fi
  done
  echo "[Launcher] no free GPU yet, sleep ${GPU_CHECK_INTERVAL}s"
  sleep "${GPU_CHECK_INTERVAL}"
done
