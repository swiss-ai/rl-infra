#!/bin/bash
# Dumps the Python and native stacks of every process of this user on the node when the
# job looks stuck, into <log dir>/stacks/<node>-<time>.txt:
# - at once when the Ray driver log reports a communication timeout (an NCCL watchdog
#   timeout, or a UCCL/DeepEP expert-dispatch timeout). The ranks that timed out abort
#   right after, but the ranks they were waiting for are still alive and stuck, so the
#   dump shows where. Lines of HANG_MATCH_MAX_CHARS (default 1000) or more are not
#   matched: they are config and environment dumps, whose UCCL_EP_CPU_TIMEOUT_SECS
#   would match;
# - when the driver log has not changed for HANG_DUMP_AFTER_S seconds (default 900), and
#   again every HANG_DUMP_EVERY_S (default 900) while it stays unchanged. Once it has been
#   idle for HANG_KILL_AFTER_S (default 2700), the node running the driver stops it after
#   the dump, so that driver_retry.sh restarts it from the latest checkpoint instead of
#   the job sitting until its time limit. Validation prints nothing until it ends, so
#   while the log's last validation marker is a start, HANG_VAL_DUMP_AFTER_S (default
#   3600) and HANG_VAL_KILL_AFTER_S (default 7200) apply instead.
# The log is checked every HANG_POLL_S seconds (default 10). Started by numa_watch.sh on
# every node, with stdout on a file in the job's log directory.
LOG_DIR="$(dirname "$(readlink -f /proc/$$/fd/1)")"
DRIVER_LOG="${LOG_DIR}/ray-driver.log"
AFTER="${HANG_DUMP_AFTER_S:-900}"
EVERY="${HANG_DUMP_EVERY_S:-900}"
KILL_AFTER="${HANG_KILL_AFTER_S:-2700}"
VAL_AFTER="${HANG_VAL_DUMP_AFTER_S:-3600}"
VAL_KILL_AFTER="${HANG_VAL_KILL_AFTER_S:-7200}"
POLL="${HANG_POLL_S:-10}"
MAX_CHARS="${HANG_MATCH_MAX_CHARS:-1000}"
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
val_checked_size=-1
while [[ ! -e "${LOG_DIR}/ENDED" ]]; do
  sleep "${POLL}"
  [[ -f "${DRIVER_LOG}" ]] || continue
  now=$(date +%s)
  size=$(stat -c %s "${DRIVER_LOG}")
  if (( size > offset )); then
    hit="$(tail -c +$(( offset + 1 )) "${DRIVER_LOG}" | awk -v max="${MAX_CHARS}" 'length($0) < max' \
      | grep -m1 -iE "${TIMEOUT_RE}")"
    offset=${size}
    if [[ -n "${hit}" ]] && (( now - last_dump >= 120 )); then
      dump "communication timeout in the driver log: $(cut -c1-300 <<<"${hit}")"
      last_dump=$(date +%s)
      continue
    fi
  fi
  idle=$(( now - $(stat -c %Y "${DRIVER_LOG}") ))
  (( idle < AFTER || now - last_dump < EVERY )) && continue
  if (( size != val_checked_size )); then
    marker="$(tac "${DRIVER_LOG}" | grep -a -m1 -oE 'Starting validation at step|Validation Results')"
    val_checked_size=${size}
  fi
  dump_after=${AFTER} kill_after=${KILL_AFTER} what="driver log idle for ${idle}s"
  if [[ "${marker}" == Starting* ]]; then
    dump_after=${VAL_AFTER} kill_after=${VAL_KILL_AFTER} what="${what} during validation"
  fi
  (( idle < dump_after )) && continue
  dump "${what}"
  last_dump=${now}
  driver='^[^ ]*python[^ ]* [^ ]*grpo[^ ]*\.py'
  if (( idle >= kill_after )) && pgrep -u "$(id -u)" -f "${driver}" >/dev/null; then
    echo "$(date +%T) driver log idle ${idle}s, stopping the driver for a retry"
    pkill -TERM -u "$(id -u)" -f "${driver}"
    sleep 60
    pkill -KILL -u "$(id -u)" -f "${driver}"
  fi
done
