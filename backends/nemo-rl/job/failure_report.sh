#!/bin/bash
# Collects what one failed driver attempt left behind into a single file,
# <job log dir>/failure-report-attempt<N>.txt: the first errors in the driver log, which
# trainer ranks hit the NCCL watchdog and which did not (the ones that did not are the ones
# to look at), the per-rank memory reports and last memory samples written by
# nemo_rl/utils/failure_forensics.py, the stacks hang_watch.sh dumped, the NCCL flight
# recorder analysis, NCCL warnings, NVIDIA Xid errors and host memory found on GPUs.
# driver_retry.sh runs it after every failed attempt; it also works by hand on an old job.
#
# Usage: failure_report.sh <job log dir> <run log dir (logger.log_dir)> [since-file] [attempt]
JOB_DIR=$1
RUN_DIR=${2:-}
SINCE=${3:-}
ATTEMPT=${4:-0}
OUT="${JOB_DIR}/failure-report-attempt${ATTEMPT}.txt"
LOG="${JOB_DIR}/ray-driver.log"
newer=()
[[ -n "${SINCE}" && -e "${SINCE}" ]] && newer=(-newer "${SINCE}")
clean() { sed 's/\x1b\[[0-9;]*m//g' "$@"; }

# The ranks a timeout leaves stuck are only dumped by hang_watch.sh after it sees the
# timeout in the log; give those dumps time to finish.
if clean "${LOG}" 2>/dev/null | grep -qiE 'watchdog caught collective operation timeout'; then
  sleep "${FAILURE_REPORT_WAIT_S:-100}"
fi

{
echo "failure report for ${JOB_DIR} (run dir ${RUN_DIR:-?}), attempt ${ATTEMPT}, written $(date +%FT%T)"

echo; echo "=== first errors in the driver log, in order (repeats of the same message left out)"
clean "${LOG}" | grep -nE 'OutOfMemoryError|out of memory|Cuda failure|DistBackendError|Watchdog caught|Fatal Python|unexpectedly exits|not registered within|Failed to startup|RayTaskError|ActorDiedError|illegal memory|NCCL error|uccl.*(error|timeout)|Traceback|Error: ' \
  | grep -viE 'nixl|triton|import_utils|fi_open_ops|GIN only|MasterConfig' \
  | awk '{k=$0; sub(/^[0-9]+:/, "", k); gsub(/[0-9]+/, "#", k); k=substr(k, 1, 160)} !seen[k]++' \
  | head -30 | cut -c1-400

echo; echo "=== NCCL watchdog timeouts per trainer rank"
timed=$(clean "${LOG}" | grep 'Watchdog caught' | grep -oE 'MegatronPolicyWorker\[rank=[0-9]+\]' | grep -oE '[0-9]+' | sort -un | tr '\n' ' ')
all=$(clean "${LOG}" | grep -oE 'MegatronPolicyWorker\[rank=[0-9]+\]' | grep -oE '[0-9]+' | sort -un | tr '\n' ' ')
if [[ -n "${timed}" ]]; then
  echo "timed out: ${timed}"
  echo "did not time out (stuck elsewhere or died first): $(comm -23 <(tr ' ' '\n' <<<"${all}" | sed '/^$/d' | sort) <(tr ' ' '\n' <<<"${timed}" | sed '/^$/d' | sort) | sort -n | tr '\n' ' ')"
else
  echo "none"
fi

F="${RUN_DIR}/forensics"
echo; echo "=== failure reports from the trainer ranks (${F})"
reports=$(find "${F}" -maxdepth 1 -name 'failure-rank*.txt' "${newer[@]}" 2>/dev/null | sort -V)
if [[ -n "${reports}" ]]; then
  for r in ${reports}; do
    echo "--- $(basename "${r}")"
    head -2 "${r}"
    grep -m1 -E '^free=' "${r}"
    grep -m1 'memory snapshot' "${r}"
  done
else
  echo "none"
fi

echo; echo "=== last memory sample per trainer rank before the failure, least free first"
for m in $(find "${F}" -maxdepth 1 -name 'mem-rank*.tsv' "${newer[@]}" 2>/dev/null); do
  echo "$(basename "${m}" .tsv) $(grep -v '^#' "${m}" | tail -1)"
done | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^free=/) f = substr($i, 6) + 0; print f "\t" $0}' | sort -n | cut -f2- | head -12

echo; echo "=== stacks dumped by hang_watch.sh (trainer processes, top frames)"
stacks=$(find "${JOB_DIR}/stacks" -name '*.txt' "${newer[@]}" 2>/dev/null | sort)
if [[ -n "${stacks}" ]]; then
  for s in ${stacks}; do
    echo "--- $(basename "${s}"): $(head -1 "${s}" | cut -c1-200)"
    awk '/^===== pid/ {show = ($0 ~ /MegatronPolicyWorker|default_worker/); n = 0; if (show) print; next}
         show && /Thread .*\(active/ {act = 1; n = 0; print "  " $0; next}
         show && act && n < 14 && /^ +/ {print "  " $0; n++}
         /^$/ {act = 0}' "${s}"
  done | cut -c1-220 | head -400
else
  echo "none"
fi

echo; echo "=== NCCL flight recorder"
fr=$(find "${RUN_DIR}" -maxdepth 1 -name 'fr-rank*' "${newer[@]}" 2>/dev/null | wc -l)
echo "${fr} rank dump(s) from this attempt in ${RUN_DIR}"
if (( fr > 0 )) && [[ -z "${FAILURE_REPORT_NO_PY:-}" ]] && python3 -c 'import torch' 2>/dev/null; then
  timeout 300 python3 -m torch.distributed.flight_recorder.fr_trace --prefix fr-rank "${RUN_DIR}" 2>&1 | head -80
fi

echo; echo "=== NCCL warnings (init noise left out)"
cat "${F}"/nccl-*.log 2>/dev/null | grep -E 'WARN|ERROR|failure' | grep -vE 'fi_open_ops|GIN only' \
  | awk '{k=$0; gsub(/^[^]]*\] /, "", k); k=substr(k, 1, 150)} !seen[k]++' | head -15 | cut -c1-300

echo; echo "=== NVIDIA Xid / kernel GPU messages"
grep -hiE 'xid|nvrm' "${JOB_DIR}"/xid-*.log 2>/dev/null | grep -vi 'read kernel buffer' | head -10

echo; echo "=== host memory that ended up on GPUs (max anon per node, GB)"
for n in "${JOB_DIR}"/numa-hbm-*.log; do
  [[ -f "${n}" ]] || continue
  awk -v n="$(basename "${n}" .log | sed 's/numa-hbm-//')" '{for (i = 1; i <= NF; i++) if ($i ~ /^anon=/) {v = substr($i, 6) + 0; if (v > m) m = v}} END {if (m >= 1) printf "%s %.1f\n", n, m}' "${n}"
done
echo "(only nodes with 1 GB or more are listed)"

echo; echo "=== Ray worker startup problems"
clean "${LOG}" | grep -E 'not registered within the timeout|Failed to startup worker' | grep -oE 'ip=[0-9.]+' | sort | uniq -c
echo "node-local Ray logs are copied to ${JOB_DIR}/raylogs/<node>/ when the job ends"
} > "${OUT}" 2>&1
echo "${OUT}"
