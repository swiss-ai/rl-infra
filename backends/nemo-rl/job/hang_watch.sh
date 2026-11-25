#!/bin/bash
# Dumps the Python and native stacks of every process of this user on the node when the
# job looks stuck, into <log dir>/stacks/<node>-<time>.txt:
# - at once when the Ray driver log reports a communication timeout (an NCCL watchdog
#   timeout, or a UCCL/DeepEP expert-dispatch timeout). The ranks that timed out abort
#   right after, but the ranks they were waiting for are still alive and stuck, so the
#   dump shows where. Lines of HANG_MATCH_MAX_CHARS (default 1000) or more are not
#   matched: they are config and environment dumps, whose UCCL_EP_CPU_TIMEOUT_SECS
#   would match;
# - at once when a UCCL rank reports waiting on a barrier for about 50 s. Healthy runs do
#   not print this. The GPU timeout traps the waiting ranks about 50 s later, while the
#   rank they wait for stays alive until the 480 s CPU timeout;
# - when <log dir>/DUMP_NOW is created or touched, on every node;
# - when the driver log has not changed for HANG_DUMP_AFTER_S seconds (default 900), and
#   again every HANG_DUMP_EVERY_S (default 900) while it stays unchanged. Once it has been
#   idle for HANG_KILL_AFTER_S (default 2700), the node running the driver stops it after
#   the dump, so that driver_retry.sh restarts it from the latest checkpoint instead of
#   the job sitting until its time limit. Validation prints nothing until it ends, so
#   while the log's last validation marker is a start, HANG_VAL_DUMP_AFTER_S (default
#   3600) and HANG_VAL_KILL_AFTER_S (default 7200) apply instead.
# The log is checked every HANG_POLL_S seconds (default 10). Started by numa_watch.sh on
# every node, with stdout on a file in the job's log directory. Each dump also samples the
# Slingshot NIC counters and the state of every thread twice, a minute or so apart: comparing
# them shows whether a NIC still moves data and whether a proxy thread still runs.
LOG_DIR="$(dirname "$(readlink -f /proc/$$/fd/1)")"
DRIVER_LOG="${LOG_DIR}/ray-driver.log"
AFTER="${HANG_DUMP_AFTER_S:-900}"
EVERY="${HANG_DUMP_EVERY_S:-900}"
KILL_AFTER="${HANG_KILL_AFTER_S:-2700}"
VAL_AFTER="${HANG_VAL_DUMP_AFTER_S:-3600}"
VAL_KILL_AFTER="${HANG_VAL_KILL_AFTER_S:-7200}"
POLL="${HANG_POLL_S:-10}"
MAX_CHARS="${HANG_MATCH_MAX_CHARS:-1000}"
TIMEOUT_RE='watchdog caught collective operation timeout|(uccl|deepep|deep_ep).*timeout|cpu recv timeout|wait_until_cmd_consumed.*waiting'
NODE="$(hostname -s)"
mkdir -p "${LOG_DIR}/stacks"

sample_host() {
  echo "===== host sample $1 at $(date +%T)"
  cxi_stat 2>&1
  grep -H . /sys/class/cxi/cxi*/device/telemetry/* 2>/dev/null
  ps -L -u "$(id -u)" -o pid,tid,psr,pcpu,time,stat,wchan:24,comm 2>&1
}

dump() {
  local out="${LOG_DIR}/stacks/${NODE}-$(date +%H%M%S).txt" tmp cmd
  tmp="$(mktemp -d)"
  echo "$(date +%T) $1, dumping stacks to ${out}"
  sample_host 1 > "${tmp}/host1"
  # Ray retitles its workers to ray::<Actor>, so their command line no longer says python.
  for pid in $(pgrep -u "$(id -u)" -f 'python|^ray::'); do
    cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | cut -c1-200)"
    [[ "${cmd}" == *compile_worker* ]] && continue
    { echo; echo "===== pid ${pid}: ${cmd}"
      timeout 90 py-spy dump --native --pid "${pid}" 2>&1; } > "${tmp}/${pid}" &
  done
  wait
  # py-spy only sees threads that run Python. The UCCL proxy threads are plain C++, and they
  # are what a stall is waiting on. gdb stops the process while it reads every thread, which
  # a healthy job can only survive if it stays under the 100 s GPU timeout, so this runs last.
  for pid in $(pgrep -u "$(id -u)" -f '^ray::MegatronPolicyWorker'); do
    { echo; echo "===== native threads of pid ${pid}"
      timeout 150 gdb -q -nx -batch -p "${pid}" -ex 'thread apply all bt 16' 2>&1 \
        | grep -vE '^(warning: |\[New LWP|\[Thread debugging|Using host libthread_db)'; } > "${tmp}/g${pid}" &
  done
  wait
  sleep 20
  sample_host 2 > "${tmp}/host2"
  { echo "$1"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv 2>&1
    cat "${tmp}"/[0-9]* "${tmp}"/g* "${tmp}/host1" "${tmp}/host2" 2>/dev/null; } > "${out}"
  rm -rf "${tmp}"
}

manual_dump=""
last_dump=0
offset=0
val_checked_size=-1
while [[ ! -e "${LOG_DIR}/ENDED" ]]; do
  sleep "${POLL}"
  [[ -f "${DRIVER_LOG}" ]] || continue
  now=$(date +%s)
  stamp="$(stat -c %Y "${LOG_DIR}/DUMP_NOW" 2>/dev/null)"
  if [[ -n "${stamp}" && "${stamp}" != "${manual_dump}" ]]; then
    manual_dump=${stamp}
    dump "requested with DUMP_NOW"
    last_dump=$(date +%s)
    continue
  fi
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
