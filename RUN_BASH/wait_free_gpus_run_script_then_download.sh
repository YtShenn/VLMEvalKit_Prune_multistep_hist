#!/usr/bin/env bash
set -euo pipefail

# Wait for free GPUs, run one target script, then rsync two folders.
#
# Usage:
#   bash RUN_BASH/wait_free_gpus_run_script_then_download.sh
#
# You can edit the config block below, or override it from command line:
#   bash RUN_BASH/wait_free_gpus_run_script_then_download.sh \
#     RUN_BASH/your_target.sh \
#     /path/to/folder1 \
#     /path/to/folder2 \
#     /path/to/local/download/root
#
# Env overrides:
#   GPU_CANDIDATES="0 1 2 3 4 5 6 7"
#   REQUIRED_GPU_COUNT=3
#   GPU_CHECK_INTERVAL=60
#   GPU_FREE_MAX_MEM_MB=1024
#   GPU_FREE_MAX_UTIL=10
#   GPU_REQUIRE_NO_COMPUTE_PROCS=1
#   FORCE_RUN_ON_GPUS=0,1,2
#   PASS_SELECTED_GPUS=1
#   TARGET_SCRIPT=RUN_BASH/your_target.sh
#   DOWNLOAD_SOURCE_DIR_1=/path/to/folder1
#   DOWNLOAD_SOURCE_DIR_2=/path/to/folder2
#   DOWNLOAD_DEST_ROOT=/path/to/local/download/root
#   DOWNLOAD_AFTER_SUCCESS_ONLY=1
#   DOWNLOAD_MISSING_OK=0
#   RSYNC_ARGS="-avh --progress"

# -------- Edit these defaults when you want to run without arguments. --------
DEFAULT_TARGET_SCRIPT="RUN_BASH/run_android_control_history_state_packet_qwen3_5_current_frame_attn_prune.sh"
DEFAULT_DOWNLOAD_SOURCE_DIR_1="OUTPUT/android_control_hist4_keep_system_prompt_state_packet_attn_prune_structured_fast_official_prune_layer3_0.8_sideattn_0828_high_task"
DEFAULT_DOWNLOAD_SOURCE_DIR_2="OUTPUT/attn_prune_debug/android_control_history_state_packet_inst_after_siedeattn_high_prune_vispred"
DEFAULT_DOWNLOAD_DEST_ROOT="${HOME}/Downloads/VLMEvalKit_Prune_multistep_hist/android_control_attn_prune"

DEFAULT_GPU_CANDIDATES="0 1 2 3 4 5 6 7"
DEFAULT_REQUIRED_GPU_COUNT=3
DEFAULT_GPU_CHECK_INTERVAL=60
DEFAULT_GPU_FREE_MAX_MEM_MB=1024
DEFAULT_GPU_FREE_MAX_UTIL=10
# ---------------------------------------------------------------------------

GPU_CANDIDATES="${GPU_CANDIDATES:-${DEFAULT_GPU_CANDIDATES}}"
REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-${DEFAULT_REQUIRED_GPU_COUNT}}"
GPU_CHECK_INTERVAL="${GPU_CHECK_INTERVAL:-${DEFAULT_GPU_CHECK_INTERVAL}}"
GPU_FREE_MAX_MEM_MB="${GPU_FREE_MAX_MEM_MB:-${DEFAULT_GPU_FREE_MAX_MEM_MB}}"
GPU_FREE_MAX_UTIL="${GPU_FREE_MAX_UTIL:-${DEFAULT_GPU_FREE_MAX_UTIL}}"
GPU_REQUIRE_NO_COMPUTE_PROCS="${GPU_REQUIRE_NO_COMPUTE_PROCS:-1}"
FORCE_RUN_ON_GPUS="${FORCE_RUN_ON_GPUS:-}"
PASS_SELECTED_GPUS="${PASS_SELECTED_GPUS:-1}"

TARGET_SCRIPT="${TARGET_SCRIPT:-${DEFAULT_TARGET_SCRIPT}}"
DOWNLOAD_SOURCE_DIR_1="${DOWNLOAD_SOURCE_DIR_1:-${DEFAULT_DOWNLOAD_SOURCE_DIR_1}}"
DOWNLOAD_SOURCE_DIR_2="${DOWNLOAD_SOURCE_DIR_2:-${DEFAULT_DOWNLOAD_SOURCE_DIR_2}}"
DOWNLOAD_DEST_ROOT="${DOWNLOAD_DEST_ROOT:-${DEFAULT_DOWNLOAD_DEST_ROOT}}"

TARGET_LOG_DIR="${TARGET_LOG_DIR:-OUTPUT/auto_gpu_launch_logs}"
RUN_BASH_BIN="${RUN_BASH_BIN:-bash}"
DOWNLOAD_AFTER_SUCCESS_ONLY="${DOWNLOAD_AFTER_SUCCESS_ONLY:-1}"
DOWNLOAD_MISSING_OK="${DOWNLOAD_MISSING_OK:-0}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"
RSYNC_ARGS_STRING="${RSYNC_ARGS:--avh --progress}"

if [[ "$#" -ge 1 ]]; then
  TARGET_SCRIPT="$1"
fi
if [[ "$#" -ge 2 ]]; then
  DOWNLOAD_SOURCE_DIR_1="$2"
fi
if [[ "$#" -ge 3 ]]; then
  DOWNLOAD_SOURCE_DIR_2="$3"
fi
if [[ "$#" -ge 4 ]]; then
  DOWNLOAD_DEST_ROOT="$4"
fi
if [[ "$#" -gt 4 ]]; then
  echo "[ERROR] Too many arguments."
  exit 1
fi

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

on_interrupt() {
  echo "[WaitRun] interrupted by user."
  exit 130
}

trap on_interrupt INT

usage_error() {
  echo "[ERROR] $1"
  echo "Usage:"
  echo "  bash $0 TARGET_SCRIPT DOWNLOAD_SOURCE_DIR_1 DOWNLOAD_SOURCE_DIR_2 DOWNLOAD_DEST_ROOT"
  exit 1
}

is_true() {
  local value="$1"
  [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "True" ]]
}

normalize_name() {
  local raw="$1"
  raw="$(basename "${raw}")"
  raw="${raw%.sh}"
  raw="${raw// /_}"
  echo "${raw}"
}

count_csv_items() {
  local value="$1"
  echo "${value}" | tr ',' '\n' | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' '
}

gpu_is_free() {
  local gpu_id="$1"
  local mem_used util_used process_lines

  mem_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -n 1 | tr -d ' ')"
  util_used="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -n 1 | tr -d ' ')"

  if [[ -z "${mem_used}" || -z "${util_used}" ]]; then
    return 1
  fi

  if [[ "${mem_used}" -gt "${GPU_FREE_MAX_MEM_MB}" || "${util_used}" -gt "${GPU_FREE_MAX_UTIL}" ]]; then
    return 1
  fi

  if is_true "${GPU_REQUIRE_NO_COMPUTE_PROCS}"; then
    process_lines="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null || true)"
    process_lines="$(echo "${process_lines}" | sed '/^[[:space:]]*$/d')"
    [[ -z "${process_lines}" ]] || return 1
  fi

  return 0
}

print_gpu_status() {
  local gpu_id line
  for gpu_id in ${GPU_CANDIDATES}; do
    line="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null || true)"
    if [[ -n "${line}" ]]; then
      echo "[GPUStatus] ${line}" >&2
    else
      echo "[GPUStatus] gpu=${gpu_id} unavailable" >&2
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

wait_for_gpus() {
  local selected_gpus forced_count

  if [[ -n "${FORCE_RUN_ON_GPUS}" ]]; then
    forced_count="$(count_csv_items "${FORCE_RUN_ON_GPUS}")"
    if [[ "${forced_count}" -lt "${REQUIRED_GPU_COUNT}" ]]; then
      echo "[ERROR] FORCE_RUN_ON_GPUS has ${forced_count} GPUs, but REQUIRED_GPU_COUNT=${REQUIRED_GPU_COUNT}."
      exit 1
    fi
    echo "[WaitRun] FORCE_RUN_ON_GPUS=${FORCE_RUN_ON_GPUS}, skip waiting." >&2
    echo "${FORCE_RUN_ON_GPUS}"
    return 0
  fi

  echo "[WaitRun] gpu_candidates=${GPU_CANDIDATES}" >&2
  echo "[WaitRun] required_gpu_count=${REQUIRED_GPU_COUNT}" >&2
  echo "[WaitRun] check_interval_s=${GPU_CHECK_INTERVAL}" >&2
  echo "[WaitRun] free_if_mem_le_mb=${GPU_FREE_MAX_MEM_MB}" >&2
  echo "[WaitRun] free_if_util_le_pct=${GPU_FREE_MAX_UTIL}" >&2
  echo "[WaitRun] require_no_compute_processes=${GPU_REQUIRE_NO_COMPUTE_PROCS}" >&2

  while true; do
    print_gpu_status
    mapfile -t FREE_GPU_LIST < <(collect_free_gpus || true)
    if [[ "${#FREE_GPU_LIST[@]}" -ge "${REQUIRED_GPU_COUNT}" ]]; then
      selected_gpus="$(IFS=,; echo "${FREE_GPU_LIST[*]:0:${REQUIRED_GPU_COUNT}}")"
      echo "[WaitRun] detected ${REQUIRED_GPU_COUNT} free GPUs: ${selected_gpus}" >&2
      echo "${selected_gpus}"
      return 0
    fi
    echo "[WaitRun] no sufficient free GPUs yet, sleep ${GPU_CHECK_INTERVAL}s" >&2
    sleep "${GPU_CHECK_INTERVAL}"
  done
}

run_target() {
  local selected_gpus="$1"
  local script_name log_file status

  mkdir -p "${TARGET_LOG_DIR}"
  script_name="$(normalize_name "${TARGET_SCRIPT}")"
  log_file="${TARGET_LOG_DIR}/${TIMESTAMP}_${script_name}_gpu${selected_gpus//,/}.log"

  echo "[WaitRun] start target_script=${TARGET_SCRIPT}"
  echo "[WaitRun] selected_gpus=${selected_gpus}"
  echo "[WaitRun] pass_selected_gpus=${PASS_SELECTED_GPUS}"
  echo "[WaitRun] log=${log_file}"

  set +e
  if is_true "${PASS_SELECTED_GPUS}"; then
    CUDA_VISIBLE_DEVICES="${selected_gpus}" "${RUN_BASH_BIN}" "${TARGET_SCRIPT}" 2>&1 | tee "${log_file}"
  else
    "${RUN_BASH_BIN}" "${TARGET_SCRIPT}" 2>&1 | tee "${log_file}"
  fi
  status=${PIPESTATUS[0]}
  set -e

  return "${status}"
}

prepare_download_dest() {
  if [[ "${DOWNLOAD_DEST_ROOT}" != *:* ]]; then
    mkdir -p "${DOWNLOAD_DEST_ROOT}"
  fi
}

download_one_dir() {
  local src="$1"
  local dst_root="$2"
  local status

  src="${src%/}"
  if [[ ! -d "${src}" ]]; then
    echo "[Download] source folder not found: ${src}"
    if is_true "${DOWNLOAD_MISSING_OK}"; then
      return 0
    fi
    return 1
  fi

  read -r -a RSYNC_ARGS_ARRAY <<< "${RSYNC_ARGS_STRING}"

  echo "[Download] rsync ${src} -> ${dst_root}/"
  set +e
  "${RSYNC_BIN}" "${RSYNC_ARGS_ARRAY[@]}" "${src}" "${dst_root}/"
  status=$?
  set -e

  if [[ "${status}" -ne 0 ]]; then
    echo "[Download] failed status=${status} src=${src}"
    return "${status}"
  fi

  echo "[Download] done ${src}"
}

download_outputs() {
  local failures=0

  prepare_download_dest

  echo "[Download] source_1=${DOWNLOAD_SOURCE_DIR_1}"
  echo "[Download] source_2=${DOWNLOAD_SOURCE_DIR_2}"
  echo "[Download] destination_root=${DOWNLOAD_DEST_ROOT}"

  download_one_dir "${DOWNLOAD_SOURCE_DIR_1}" "${DOWNLOAD_DEST_ROOT}" || failures=$((failures + 1))
  download_one_dir "${DOWNLOAD_SOURCE_DIR_2}" "${DOWNLOAD_DEST_ROOT}" || failures=$((failures + 1))

  if [[ "${failures}" -gt 0 ]]; then
    echo "[Download] finished with failures=${failures}"
    return 1
  fi

  echo "[Download] all folders synced successfully."
}

[[ -n "${TARGET_SCRIPT}" ]] || usage_error "TARGET_SCRIPT is required."
[[ -n "${DOWNLOAD_SOURCE_DIR_1}" ]] || usage_error "DOWNLOAD_SOURCE_DIR_1 is required."
[[ -n "${DOWNLOAD_SOURCE_DIR_2}" ]] || usage_error "DOWNLOAD_SOURCE_DIR_2 is required."
[[ -n "${DOWNLOAD_DEST_ROOT}" ]] || usage_error "DOWNLOAD_DEST_ROOT is required."
[[ -f "${TARGET_SCRIPT}" ]] || usage_error "Target script not found: ${TARGET_SCRIPT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found in PATH."
  exit 1
fi

if ! command -v "${RSYNC_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] ${RSYNC_BIN} not found in PATH."
  exit 1
fi

selected_gpus="$(wait_for_gpus | tail -n 1)"

set +e
run_target "${selected_gpus}"
target_status=$?
set -e

if [[ "${target_status}" -eq 0 ]]; then
  echo "[WaitRun] target script finished successfully."
elif [[ "${target_status}" -eq 130 ]]; then
  echo "[WaitRun] target script interrupted by user."
  exit 130
else
  echo "[WaitRun] target script failed status=${target_status}."
  if is_true "${DOWNLOAD_AFTER_SUCCESS_ONLY}"; then
    echo "[WaitRun] download_after_success_only=1, skip download."
    exit "${target_status}"
  fi
fi

set +e
download_outputs
download_status=$?
set -e

if [[ "${target_status}" -ne 0 ]]; then
  exit "${target_status}"
fi

exit "${download_status}"
