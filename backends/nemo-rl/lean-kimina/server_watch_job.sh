#!/bin/bash
# The Kimina server job that train.sh submits: server_job.sh, unchanged, plus a watch on the training job it
# serves. Each action of the watch is logged with "[watch ...]" in this job's output:
#   1. When the training job has left the queue, it stops the server, so this job ends by itself.
#   2. If the training runs but its Ray head is not up HEAD_LIMIT seconds (default 600; a good start needs about
#      60) after THIS job started, it cancels the training job.
#   3. If the server ends by itself while the training is still queued or running, it cancels the training job:
#      without the verifier the run is worth nothing.
# Why 2: in run 3949635 (Clariden, 2026-11-16) the container of the head node never started, ray.sub waits for
# the Ray head without a limit, and 16 nodes idled for the whole 1:30 h.
# Needs TRAIN_JOB, TRAIN_LOGS (the training's <jobid>-logs folder) and what server_job.sh needs.
set -u
: "${TRAIN_JOB:?}" "${TRAIN_LOGS:?}" "${LEAN_KIMINA_DIR:?}"
HEAD_LIMIT="${HEAD_LIMIT:-600}"
POLL="${WATCH_POLL:-20}"
say() { echo "[watch $(date '+%F %T')] $*"; }

bash "${LEAN_KIMINA_DIR}/server_job.sh" &
SERVER=$!
# server_job.sh's own EXIT trap stops the Kimina server and its REPLs and removes the copy in /dev/shm.
stop_server() { kill -TERM "${SERVER}" 2> /dev/null; wait "${SERVER}" 2> /dev/null; }
trap 'say "signal: stopping the server"; stop_server; exit 143' TERM INT

# "RUNNING", "PENDING", ... while the job is in the queue; "gone" when squeue answers that it is not;
# "unknown" when squeue itself failed (never taken as "gone").
train_state() {
    local out rc
    out="$(squeue -h -j "${TRAIN_JOB}" -o %T 2>&1)"; rc=$?
    if [[ ${rc} -eq 0 && -n "${out}" ]]; then echo "${out%%$'\n'*}"
    elif [[ ${rc} -eq 0 || "${out}" == *"Invalid job id"* ]]; then echo gone
    else echo unknown
    fi
}

say "serving training ${TRAIN_JOB}; Ray head limit ${HEAD_LIMIT} s"
t0=${SECONDS}; gone=0; head_done=0; server_rc=""
while true; do
    sleep "${POLL}" & wait $!
    state="$(train_state)"
    if [[ "${state}" == gone ]]; then gone=$((gone + 1)); else gone=0; fi
    if ! kill -0 "${SERVER}" 2> /dev/null; then
        wait "${SERVER}"; server_rc=$?
        say "the server ended by itself (exit ${server_rc}); training is ${state}"
        if [[ "${state}" != gone && "${state}" != unknown ]]; then
            say "cancelling training ${TRAIN_JOB}: it has no verifier"
            scancel "${TRAIN_JOB}"
        fi
        exit "${server_rc}"
    fi
    if (( gone >= 3 )); then
        say "training ${TRAIN_JOB} has left the queue: stopping the server"
        break
    fi
    if (( head_done == 0 )); then
        if [[ -f "${TRAIN_LOGS}/STARTED_RAY_HEAD" ]]; then
            head_done=1; say "the Ray head of ${TRAIN_JOB} is up ($(( SECONDS - t0 )) s after this job started)"
        elif [[ "${state}" == RUNNING ]] && (( SECONDS - t0 > HEAD_LIMIT )); then
            head_done=1
            say "no Ray head ${HEAD_LIMIT} s after this job started: cancelling training ${TRAIN_JOB}"
            scancel "${TRAIN_JOB}"
        fi
    fi
done
stop_server
say "done"
