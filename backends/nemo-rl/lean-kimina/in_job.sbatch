#!/bin/bash
# Batch script of a Lean training job with its Kimina server inside the job (train.sh, the default SERVER=in-job;
# launch.sh submits it as BATCH_SCRIPT). It runs on the job's first node, the Ray head:
#   1. starts server_job.sh there, in the background, outside the container (as the sampler runs Kimina next to
#      vLLM); the training reads its address and key at job start through endpoint.sh, as before;
#   2. runs NeMo-RL's ray.sub (RAY_SUB) unchanged, and ends with its exit status;
#   3. restarts a server that dies, up to KIMINA_RESTARTS times, under the same address and key; when none is
#      left it cancels the job, which is worth nothing without its verifier;
#   4. cancels the job if the Ray head is not up KIMINA_HEAD_LIMIT seconds after the job started (in run 3949635
#      the head's container never started, and ray.sub waits for it without a limit).
# The server's memory stays on the Grace CPUs' NUMA nodes (KIMINA_MEMBIND), off the GPUs' memory, and it runs
# at a lower CPU priority than the training. It is small, because the first node is also a trainer and the
# training's Ray head: in run 3948553 that node bottomed out at 80 GB free of the job's 450 GB per node
# (trainer nodes at 104-142 GB, vLLM nodes above 350 GB). A warm REPL holds about 2.2 GB at peak and the Lean
# copy in /dev/shm 9 GB, so 16 REPLs (KIMINA_REPLS) take about 44 GB. A requeued job starts all of this again.
# Each action is logged with "[kimina ...]" in the job's output.
# The #SBATCH lines are ray.sub's; launch.sh gives the rest on the command line.
#SBATCH --exclusive
#SBATCH --dependency=singleton
set -uo pipefail
: "${RAY_SUB:?}" "${KIMINA_DIR:?}" "${KIMINA_RL_INFRA:?}" "${KIMINA_ENDPOINT_FILE:?}" "${KIMINA_HEADER_ROWS:?}"
REPLS="${KIMINA_REPLS:-16}"
RESTARTS="${KIMINA_RESTARTS:-3}"
HEAD_LIMIT="${KIMINA_HEAD_LIMIT:-600}"
MEMBIND="${KIMINA_MEMBIND-0-3}"
LOGS="$(dirname "${KIMINA_ENDPOINT_FILE}")/logs"
# ray.sub's own log folder, where it writes STARTED_RAY_HEAD.
if [[ -n "${SLURM_RESTART_COUNT:-}" ]]; then
    RAY_LOGS="${BASE_LOG_DIR:-${SLURM_SUBMIT_DIR}}/${SLURM_JOB_ID}-${SLURM_RESTART_COUNT}-logs"
else
    RAY_LOGS="${BASE_LOG_DIR:-${SLURM_SUBMIT_DIR}}/${SLURM_JOB_ID}-logs"
fi
TAG="${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT:-0}"
mkdir -p "${LOGS}"
say() { echo "[kimina $(date '+%F %T')] $*"; }

# numactl keeps the server's memory, and the Lean copy in /dev/shm it writes, on the CPU NUMA nodes.
membind=()
if [[ -n "${MEMBIND}" ]] && command -v numactl > /dev/null && numactl --membind="${MEMBIND}" true 2> /dev/null; then
    membind=(numactl --membind="${MEMBIND}")
fi

# The server, restarted while the job runs. PORT, MAX_REPLS, ... are server_job.sh's names; they are set for it
# alone (ray.sub reads PORT as Ray's own port).
serve() {
    local n=0 rc
    while true; do
        say "server start ${n} on $(hostname):${KIMINA_PORT:-1120}, ${REPLS} REPLs, log ${LOGS}/${TAG}-kimina-${n}.out"
        env LEAN_KIMINA_DIR="${KIMINA_DIR}" ROOT="${KIMINA_RL_INFRA}" ENDPOINT_FILE="${KIMINA_ENDPOINT_FILE}" \
            MAX_REPLS="${REPLS}" PORT="${KIMINA_PORT:-1120}" HEADER_ROWS="${KIMINA_HEADER_ROWS}" KEEP_KEY=1 \
            SERVER_LOG="${LOGS}/${TAG}-server-${n}.log" \
            ${membind[@]+"${membind[@]}"} nice -n "${KIMINA_NICE:-5}" bash "${KIMINA_DIR}/server_job.sh" \
            > "${LOGS}/${TAG}-kimina-${n}.out" 2>&1 &
        wait $!; rc=$?
        n=$((n + 1))
        if (( n > RESTARTS )); then
            say "the server ended (exit ${rc}) and no restart is left: cancelling job ${SLURM_JOB_ID}, it has no verifier"
            scancel "${SLURM_JOB_ID}"
            return "${rc}"
        fi
        say "the server ended (exit ${rc}); restart ${n} of ${RESTARTS} in 10 s"
        sleep 10
    done
}

# The Ray head check.
watch_head() {
    local t=0
    while (( t < HEAD_LIMIT )); do
        [[ -f "${RAY_LOGS}/STARTED_RAY_HEAD" ]] && { say "the Ray head is up after ${t} s"; return 0; }
        sleep 10; t=$((t + 10))
    done
    [[ -f "${RAY_LOGS}/STARTED_RAY_HEAD" ]] && return 0
    say "no Ray head (${RAY_LOGS}/STARTED_RAY_HEAD) ${HEAD_LIMIT} s after the start: cancelling job ${SLURM_JOB_ID}"
    scancel "${SLURM_JOB_ID}"
}

# Process groups of their own, so stopping them also stops the server's REPLs and the sleeps.
set -m
serve & SERVE_PID=$!
watch_head & WATCH_PID=$!
set +m
stop() {
    kill -TERM -- "-${WATCH_PID}" 2> /dev/null
    kill -TERM -- "-${SERVE_PID}" 2> /dev/null
    wait "${SERVE_PID}" 2> /dev/null
}
RAY_PID=""
trap 'say "signal: stopping"; [[ -n "${RAY_PID}" ]] && kill -TERM "${RAY_PID}" 2> /dev/null; stop; wait 2> /dev/null; exit 143' TERM INT

# The job's real end: with --time-min Slurm sets the time limit only when the job starts, and a requeued job
# gets a new one. driver_retry.sh saves the last checkpoint NRL_SAVE_MARGIN_S before it (ray.sub passes the
# environment into the container).
end=$(SLURM_TIME_FORMAT=%s squeue -h -j "${SLURM_JOB_ID}" -o %e 2> /dev/null | head -n 1)
if [[ "${end}" =~ ^[0-9]+$ ]]; then
    export NRL_JOB_END_EPOCH="${end}"
    say "the job ends $(date -d "@${end}" '+%F %T') ($(( (end - $(date +%s)) / 60 )) min from now)"
else
    say "could not read the job's end time (squeue: '${end}'); the save deadline set at submission stays"
fi

bash "${RAY_SUB}" &
RAY_PID=$!
wait "${RAY_PID}"; rc=$?
say "ray.sub ended with ${rc}: stopping the server"
stop
exit "${rc}"
