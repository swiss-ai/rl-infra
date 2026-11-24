#!/bin/bash
# Submit the 200-step run as a chain of N_JOBS training jobs, each followed by a small guard job:
#   J1 -> G1 -> J2 -> G2 -> ... -> JN -> GN
# Every training job J<k> is run-200.sh with the same RUN, OUT and Lean, so it resumes from the run's newest
# checkpoint. On preemptable, Slurm requeues a preempted J<k> by itself (--requeue, at most 5 times per job,
# MaxBatchRequeue); G<k> (partition xfer, 1 CPU, --dependency=afterany:J<k>) runs only once J<k> has ended for good
# (time limit, more than 5 preemptions, failure, completion). It lets J<k+1> start (exit 0) only if the run is not
# done, no STOP file exists, and the run made progress (one start without progress is forgiven); otherwise it
# cancels the rest of the chain. J<k+1> waits for G<k> with --dependency=afterok:G<k>,singleton.
#   DRY_RUN=1 bash start.sh   run-200.sh's dry run and the planned guard commands; submits nothing
#   bash start.sh             submits the chain
# Settings: PARTITION (default: normal if Slurm expects 12 nodes to start there within NORMAL_WAIT_MAX seconds,
# default 3600, else preemptable), JOB_TIME (24:00:00 on preemptable, 12:00:00 on normal), TIME_MIN (default
# 06:00:00; empty: off), N_JOBS (10 on preemptable, 8 on normal).
# TIME_MIN is sbatch's --time-min: Slurm may start a job early in a gap of the schedule that is shorter than
# JOB_TIME but at least TIME_MIN, and cuts its time limit to the gap. In the 14 days to 2026-11-24, 9-16-node
# preemptable jobs with limits up to 8 h started after a median of 0.0-0.5 h, those with 24 h after 1.8 h (p75 17 h).
# in_job.sbatch reads the real end at every start, and the driver saves its last checkpoint 30 min before it.
# Shorter jobs mean more jobs per run, hence the larger N_JOBS.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/settings.sh"
[[ "${HERE}" == "${CHAIN}" ]] || { echo "ERROR: run start.sh from ${CHAIN} (the chain's folder), not ${HERE}" >&2; exit 1; }
DRY=${DRY_RUN:-0}

# One chain at a time.
active=$(squeue -h -u "${USER}" -n "${TRAIN_NAME},${GUARD_NAME}" -o %i | tr '\n' ' ')
[[ -z "${active// /}" ]] || { echo "ERROR: jobs of this run are still in the queue: ${active}(stop.sh cancels them)" >&2; exit 1; }
[[ ! -e "${CHAIN}/DONE" ]] || { echo "ERROR: ${CHAIN}/DONE exists: the run is complete" >&2; exit 1; }
[[ ! -e "${CHAIN}/STOP" ]] || { echo "ERROR: ${CHAIN}/STOP exists: remove it to start a new chain" >&2; exit 1; }
step=$(latest_step)
(( step < MAX_STEPS )) || { echo "ERROR: the run already has step ${step}" >&2; exit 1; }

# Partition: normal (never preempted, 12 h) if it is expected to start soon, else preemptable (24 h, requeued).
if [[ -z "${PARTITION:-}" ]]; then
    est=$(cd "${SCRATCH}" && sbatch -A infra01 --test-only --partition=normal --nodes=12 --exclusive --time=12:00:00 --wrap=true 2>&1 \
          | grep -oE 'to start at [^ ]+' | awk '{print $4}') || true
    # Slurm prints 2026-11-23T10:25:37, or only 10:25:37 for today (a past time of day means tomorrow).
    wait_s=$(python3 -c '
import sys, datetime as d
s, now = sys.argv[1], d.datetime.now()
try:
    t = d.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
except ValueError:
    try:
        t = d.datetime.combine(now.date(), d.datetime.strptime(s, "%H:%M:%S").time())
        if t < now - d.timedelta(minutes=5):
            t += d.timedelta(days=1)
    except ValueError:
        print(10**9); sys.exit()
print(max(0, int((t - now).total_seconds())))' "${est:-none}")
    if (( wait_s <= ${NORMAL_WAIT_MAX:-3600} )); then PARTITION=normal; else PARTITION=preemptable; fi
    echo "normal: expected start ${est:-unknown} (in ${wait_s} s) -> ${PARTITION}"
fi
case "${PARTITION}" in
    preemptable) JOB_TIME=${JOB_TIME:-24:00:00}; N_JOBS=${N_JOBS:-10} ;;
    normal)      JOB_TIME=${JOB_TIME:-12:00:00}; N_JOBS=${N_JOBS:-8} ;;
    *) echo "ERROR: PARTITION is preemptable or normal" >&2; exit 1 ;;
esac
TIME_MIN=${TIME_MIN-06:00:00}
EXTRA="--kill-on-invalid-dep=yes${TIME_MIN:+ --time-min=${TIME_MIN}}"
echo "chain: ${N_JOBS} x (${PARTITION}, ${JOB_TIME}${TIME_MIN:+, at least ${TIME_MIN}}), RUN ${RUN}, OUT ${OUT}, Lean ${LEAN_ROOT}, newest step ${step}"

guard_args() {  # guard_args <k> <training job>: sets G_ARGS, the guard's sbatch arguments
    G_ARGS=(--parsable --account=infra01 --partition=xfer --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G
        --time=00:10:00 --job-name="${GUARD_NAME}" --dependency="afterany:$2" --kill-on-invalid-dep=yes
        --output="${CHAIN}/logs/%j-guard.out" --export="ALL,CHAIN_DIR=${CHAIN},CHAIN_K=$1,TRAIN_JOB=$2" "${CHAIN}/guard.sh")
}
guard_cmd() { guard_args "$@"; echo "sbatch ${G_ARGS[*]}"; }
train() {  # train <dependency or empty>: run-200.sh; prints the training job's ID
    local out
    out=$(cd "${RL_INFRA}" && env RUN="${RUN}" PARTITION="${PARTITION}" JOB_TIME="${JOB_TIME}" DEPENDENCY="$1" \
          SBATCH_EXTRA="${EXTRA}" DRY_RUN="${DRY}" bash "${RUN200}" 2>&1 | tee -a "${CHAIN}/logs/submit.log") \
        || { echo "ERROR: run-200.sh failed (${CHAIN}/logs/submit.log)" >&2; return 1; }
    if [[ "${DRY}" == 1 ]]; then echo "${out}" | tail -n 25 >&2; echo "<J>"; return 0; fi
    grep -oE "^Run .* is job [0-9]+\.$" <<< "${out}" | grep -oE '[0-9]+\.$' | tr -d . \
        || { echo "ERROR: no job ID in run-200.sh's output (${CHAIN}/logs/submit.log)" >&2; return 1; }
}

mkdir -p "${CHAIN}/logs"
if [[ "${DRY}" != 1 ]]; then
    header=$(printf '# chain started %s: %s jobs, %s, %s, time-min %s, from step %s' "$(date '+%F %T')" "${N_JOBS}" "${PARTITION}" "${JOB_TIME}" "${TIME_MIN:-none}" "${step}")
    echo "${header}" >> "${CHAIN}/chain.tsv"
    echo "${header}" >> "${CHAIN}/guard.tsv"
fi
dep="" ; submitted=()
for k in $(seq 1 "${N_JOBS}"); do
    if [[ "${DRY}" == 1 ]]; then
        # run-200.sh's dry run once; the other jobs differ only in DEPENDENCY.
        if [[ $k -eq 1 ]]; then train "" > /dev/null; fi
        echo "J${k}: run-200.sh with PARTITION=${PARTITION} JOB_TIME=${JOB_TIME} DEPENDENCY='${dep}' SBATCH_EXTRA='${EXTRA}'"
        echo "G${k}: $(guard_cmd "${k}" "<J${k}>")"
        dep="afterok:<G${k}>,singleton"
        continue
    fi
    j=$(train "${dep}") || { echo "ERROR: chain stops at job ${k}; submitted so far: ${submitted[*]:-none}" >&2; exit 1; }
    guard_args "${k}" "${j}"
    g=$(sbatch "${G_ARGS[@]}") || { echo "ERROR: guard ${k} not submitted; cancelling training ${j}" >&2; scancel "${j}"; exit 1; }
    printf '%s\t%s\t%s\t%s\t%s\n' "${k}" "${j}" "${g}" "${PARTITION}" "$(date '+%F %T')" >> "${CHAIN}/chain.tsv"
    submitted+=("${j}" "${g}")
    echo "J${k} = ${j} (${PARTITION}${dep:+, after ${dep}}), G${k} = ${g}"
    dep="afterok:${g},singleton"
done
[[ "${DRY}" == 1 ]] && { echo "dry run: nothing submitted"; exit 0; }
echo "chain submitted: ${submitted[*]}; status: bash ${CHAIN}/status.sh; stop: bash ${CHAIN}/stop.sh"
