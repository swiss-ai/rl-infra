#!/bin/bash
# Dumps the Python and native stacks of every process of this user on the node when the
# job looks stuck: the Ray driver log has not changed for HANG_DUMP_AFTER_S seconds
# (default 900). Writes <log dir>/stacks/<node>-<time>.txt, then again every
# HANG_DUMP_EVERY_S (default 900) while the log stays unchanged. Once the log has been idle
# for HANG_KILL_AFTER_S (default 2700), the node running the driver stops it after the dump,
# so that driver_retry.sh restarts it from the latest checkpoint instead of the job sitting
# until its time limit. Started by numa_watch.sh on every node, with stdout on a file in the
# job's log directory.
LOG_DIR="$(dirname "$(readlink -f /proc/$$/fd/1)")"
DRIVER_LOG="${LOG_DIR}/ray-driver.log"
AFTER="${HANG_DUMP_AFTER_S:-900}"
EVERY="${HANG_DUMP_EVERY_S:-900}"
KILL_AFTER="${HANG_KILL_AFTER_S:-2700}"
NODE="$(hostname -s)"
mkdir -p "${LOG_DIR}/stacks"
last_dump=0
while [[ ! -e "${LOG_DIR}/ENDED" ]]; do
  sleep 60
  [[ -f "${DRIVER_LOG}" ]] || continue
  now=$(date +%s)
  idle=$(( now - $(stat -c %Y "${DRIVER_LOG}") ))
  (( idle < AFTER || now - last_dump < EVERY )) && continue
  out="${LOG_DIR}/stacks/${NODE}-$(date +%H%M%S).txt"
  echo "$(date +%T) driver log idle ${idle}s, dumping stacks to ${out}"
  {
    echo "driver log idle for ${idle}s"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv 2>&1
    for pid in $(pgrep -u "$(id -u)" -f python); do
      echo; echo "===== pid ${pid}: $(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | cut -c1-200)"
      timeout 120 py-spy dump --native --pid "${pid}" 2>&1
    done
  } > "${out}" 2>&1
  last_dump=${now}
  driver='^[^ ]*python[^ ]* [^ ]*grpo[^ ]*\.py'
  if (( idle >= KILL_AFTER )) && pgrep -u "$(id -u)" -f "${driver}" >/dev/null; then
    echo "$(date +%T) driver log idle ${idle}s, stopping the driver for a retry"
    pkill -TERM -u "$(id -u)" -f "${driver}"
    sleep 60
    pkill -KILL -u "$(id -u)" -f "${driver}"
  fi
done
