#!/bin/bash
# Dumps the Python and native stacks of every process of this user on the node when the
# job looks stuck, into <log dir>/stacks/<node>-<time>.txt:
# - at once when the Ray driver log reports a communication timeout (an NCCL watchdog
#   timeout, or a UCCL/DeepEP expert-dispatch timeout). The ranks that timed out abort
#   right after, but the ranks they were waiting for are still alive and stuck, so the
#   dump shows where;
# - when the driver log has not changed for HANG_DUMP_AFTER_S seconds (default 900), and
#   again every HANG_DUMP_EVERY_S (default 900) while it stays unchanged. Once it has been
#   idle for HANG_KILL_AFTER_S (default 2700), the node running the driver stops it after
#   the dump, so that driver_retry.sh restarts it from the latest checkpoint instead of
#   the job sitting until its time limit.
# The log is checked every HANG_POLL_S seconds (default 10). Started by numa_watch.sh on
# every node, with stdout on a file in the job's log directory.
LOG_DIR="$(dirname "$(readlink -f /proc/$$/fd/1)")"
DRIVER_LOG="${LOG_DIR}/ray-driver.log"
AFTER="${HANG_DUMP_AFTER_S:-900}"
EVERY="${HANG_DUMP_EVERY_S:-900}"
KILL_AFTER="${HANG_KILL_AFTER_S:-2700}"
POLL="${HANG_POLL_S:-10}"
TIMEOUT_RE='watchdog caught collective operation timeout|(uccl|deepep|deep_ep).*timeout|cpu recv timeout'
NODE="$(hostname -s)"
mkdir -p "${LOG_DIR}/stacks"

dump() {
  local out="${LOG_DIR}/stacks/${NODE}-$(date +%H%M%S).txt" tmp
  tmp="$(mktemp -d)"
  echo "$(date +%T) $1, dumping stacks to ${out}"
  for pid in $(pgrep -u "$(id -u)" -f python); do
    { echo; echo "===== pid ${pid}: $(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | cut -c1-200)"
      timeout 90 py-spy dump --native --pid "${pid}" 2>&1; } > "${tmp}/${pid}" &
  done
  wait
  { echo "$1"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv 2>&1
    cat "${tmp}"/* 2>/dev/null; } > "${out}"
  rm -rf "${tmp}"
}

last_dump=0
offset=0
while [[ ! -e "${LOG_DIR}/ENDED" ]]; do
  sleep "${POLL}"
  [[ -f "${DRIVER_LOG}" ]] || continue
  now=$(date +%s)
  size=$(stat -c %s "${DRIVER_LOG}")
  if (( size > offset )); then
    hit="$(tail -c +$(( offset + 1 )) "${DRIVER_LOG}" | grep -m1 -iE "${TIMEOUT_RE}")"
    offset=${size}
    if [[ -n "${hit}" ]] && (( now - last_dump >= 120 )); then
      dump "communication timeout in the driver log: $(cut -c1-300 <<<"${hit}")"
      last_dump=$(date +%s)
      continue
    fi
  fi
  idle=$(( now - $(stat -c %Y "${DRIVER_LOG}") ))
  (( idle < AFTER || now - last_dump < EVERY )) && continue
  dump "driver log idle for ${idle}s"
  last_dump=${now}
  driver='^[^ ]*python[^ ]* [^ ]*grpo[^ ]*\.py'
  if (( idle >= KILL_AFTER )) && pgrep -u "$(id -u)" -f "${driver}" >/dev/null; then
    echo "$(date +%T) driver log idle ${idle}s, stopping the driver for a retry"
    pkill -TERM -u "$(id -u)" -f "${driver}"
    sleep 60
    pkill -KILL -u "$(id -u)" -f "${driver}"
  fi
done
