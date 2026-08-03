#!/usr/bin/env bash
set -euo pipefail

# Run multiple rsync copy jobs sequentially.
# You can either:
# 1. Edit DEFAULT_TASKS below, then run:
#      bash RUN_BASH/run_rsync_sequential.sh
# 2. Or override from command line with src/dst pairs:
#      bash RUN_BASH/run_rsync_sequential.sh /src/a /dst/a /src/b /dst/b
#
# Optional env vars:
#   RSYNC_BIN=rsync                rsync executable
#   RSYNC_ARGS="-av"               extra rsync arguments
#   RSYNC_STOP_ON_ERROR=1          stop immediately on first failure (default)

DEFAULT_TASKS=(
  "/mnt/storage2/users/ytshen_data/GUI-Actor-Data::: /mnt/storage2/public_data/Datasets"
)

RSYNC_BIN="${RSYNC_BIN:-rsync}"
RSYNC_ARGS_STRING="${RSYNC_ARGS:--av}"
STOP_ON_ERROR="${RSYNC_STOP_ON_ERROR:-1}"

on_interrupt() {
  echo "[RsyncSeq] interrupted by user (SIGINT)."
  exit 130
}

trap on_interrupt INT

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s\n' "$s"
}

TASKS=()
if [[ "$#" -gt 0 ]]; then
  if (( $# % 2 != 0 )); then
    echo "[ERROR] Command line args must be provided as src/dst pairs."
    echo "Example:"
    echo "  bash $0 /src/a /dst/a /src/b /dst/b"
    exit 1
  fi

  while [[ "$#" -gt 0 ]]; do
    src="$1"
    dst="$2"
    TASKS+=("${src}:::${dst}")
    shift 2
  done
else
  TASKS=("${DEFAULT_TASKS[@]}")
fi

if [[ "${#TASKS[@]}" -lt 1 ]]; then
  echo "[ERROR] No rsync tasks configured."
  exit 1
fi

read -r -a RSYNC_ARGS_ARRAY <<< "${RSYNC_ARGS_STRING}"

failures=0
index=0
total="${#TASKS[@]}"

echo "[RsyncSeq] configured tasks:"
for task in "${TASKS[@]}"; do
  src="$(trim "${task%%:::*}")"
  dst="$(trim "${task#*:::}")"
  echo "  - ${src} -> ${dst}"
done

for task in "${TASKS[@]}"; do
  index=$((index + 1))

  src="$(trim "${task%%:::*}")"
  dst="$(trim "${task#*:::}")"

  if [[ -z "${src}" || -z "${dst}" ]]; then
    echo "[ERROR] Invalid task at index ${index}: ${task}"
    exit 1
  fi

  if [[ ! -e "${src}" ]]; then
    echo "[ERROR] Source not found: ${src}"
    exit 1
  fi

  echo "[RsyncSeq] (${index}/${total}) start ${src} -> ${dst}"

  set +e
  "${RSYNC_BIN}" "${RSYNC_ARGS_ARRAY[@]}" "${src}" "${dst}"
  status=$?
  set -e

  if [[ "${status}" -eq 0 ]]; then
    echo "[RsyncSeq] (${index}/${total}) done ${src} -> ${dst}"
  elif [[ "${status}" -eq 130 ]]; then
    echo "[RsyncSeq] (${index}/${total}) interrupted by user status=130"
    exit 130
  else
    failures=$((failures + 1))
    echo "[RsyncSeq] (${index}/${total}) failed status=${status} src=${src} dst=${dst}"
    if [[ "${STOP_ON_ERROR}" == "1" || "${STOP_ON_ERROR}" == "true" || "${STOP_ON_ERROR}" == "True" ]]; then
      echo "[RsyncSeq] stop_on_error=1, aborting."
      exit "${status}"
    fi
  fi
done

if [[ "${failures}" -gt 0 ]]; then
  echo "[RsyncSeq] finished with failures=${failures}"
  exit 1
fi

echo "[RsyncSeq] all copy tasks finished successfully."
