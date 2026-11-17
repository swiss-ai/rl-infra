#!/bin/bash
# Train on the Lean task (envs/lean-kimina-sft.yaml) with a Kimina server of its own, through run.sh.
#   DRY_RUN=1 bash backends/nemo-rl/lean-kimina/train.sh [overrides...]   setup, preflight and the sbatch command(s)
#   bash backends/nemo-rl/lean-kimina/train.sh [overrides...]             submits
# Overrides go to run.sh as usual (e.g. grpo.max_num_steps=3). See RUNBOOK.md for a first run.
#
# SERVER=in-job (default): the server runs inside the training job, on its first node (the Ray head), started by
# in_job.sbatch before ray.sub; it lives and ends with the job, also across a requeue. No extra node, no debug
# job. The training reads the server's address and key at job start through endpoint.sh, which waits for a live
# server.
# SERVER=debug: the server is a 1-node job of its own on debug (server_watch_job.sh), submitted after the
# training with --dependency=after:<training>; it stops when the training is over, and cancels a training whose
# Ray head is not up after HEAD_LIMIT seconds or whose server died. Debug limits it to 1:30 h and one job per user.
#
# Settings (environment):
#   RECIPE           run.sh's recipe; default the task's own. For an SFT checkpoint, a recipe that names its
#                    sft_export, e.g. recipes/apertus2-chonk-sftv02-18315-vllm-adamw-code.yaml.
#   RUN              run name; default lean-kimina-<date>-<time>. The server's endpoint and logs go to
#                    $SCRATCH/lean-kimina/runs/<RUN>/.
#   PARTITION        training partition, default normal (never preempted, at most 12 h). On preemptable (at most
#                    1 day) an in-job run is requeued when preempted and resumes from its last checkpoint.
#   JOB_TIME         training time limit; default 12:00:00 in-job, 01:20:00 with SERVER=debug. In-job, the run
#                    saves by JOB_TIME - 30 min (checkpointing.checkpoint_must_save_by; an override of it wins).
#   MAX_REPLS        the server's REPLs (and the Gym's requests in flight); default 16 in-job, 128 with
#                    SERVER=debug. About 2.2 GB each at peak. In-job they share the first node with a trainer and the
#                    Ray head, which had only ~80 GB free at its lowest (run 3948553); see in_job.sbatch.
#                    KIMINA_MAX_REPL_MEM caps one REPL (default 8G in-job).
#   HEAD_LIMIT       seconds the training's Ray head may take to come up before the job is cancelled, default 600.
#   In-job only: KIMINA_PORT (default 1120: ray.sub keeps 1150-8999 for itself), KIMINA_RESTARTS (default 3),
#                KIMINA_MEMBIND (default 0-3, the CPU NUMA nodes; empty: none), KIMINA_NICE (default 5).
#   SERVER=debug only: SERVER_PARTITION (default debug) and SERVER_TIME (default 01:30:00), which must cover
#                JOB_TIME. While another debug job of yours runs, the server waits, and the training waits for it in
#                endpoint.sh for at most 30 min (KIMINA_ENDPOINT_WAIT), then fails.
set -euo pipefail
LEAN_KIMINA_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${LEAN_KIMINA_DIR}/../../.." && pwd)"
# sbatch passes the whole environment into the jobs, and ray.sub prints most of it: keep secrets out. W&B reads
# its key from WANDB_KEY_FILE (see site.env.example).
unset WANDB_API_KEY HF_TOKEN OPENAI_API_KEY
source "${ROOT}/sites/load.sh" && rl_infra_load "${LEAN_KIMINA_DIR}/env"
source "${LEAN_KIMINA_DIR}/pin"
source "${LEAN_KIMINA_DIR}/kimina.sh"
[[ -x "$(kimina_repl_bin)" ]] || { echo "ERROR: no Kimina install in ${KIMINA_INSTALL}; run setup.sh first" >&2; exit 1; }
rows="${HEADER_ROWS:-${LEAN_DATA}/data/train.v434.jsonl}"
[[ -s "${rows}" ]] || { echo "ERROR: no training rows ${rows}; run setup-data.sh first" >&2; exit 1; }

SERVER="${SERVER:-in-job}"
[[ "${SERVER}" == in-job || "${SERVER}" == debug ]] || { echo "ERROR: SERVER is in-job or debug, not ${SERVER}" >&2; exit 1; }
RUN="${RUN:-lean-kimina-$(date +%Y%m%d-%H%M%S)}"
if [[ "${SERVER}" == in-job ]]; then
    JOB_TIME="${JOB_TIME:-12:00:00}"; MAX_REPLS="${MAX_REPLS:-16}"
else
    JOB_TIME="${JOB_TIME:-01:20:00}"; MAX_REPLS="${MAX_REPLS:-128}"
fi
SERVER_TIME="${SERVER_TIME:-01:30:00}"
SERVER_PARTITION="${SERVER_PARTITION:-debug}"
DIR="${SCRATCH}/lean-kimina/runs/${RUN}"
ENDPOINT="${DIR}/endpoint"

# [D-]HH:MM:SS in seconds; empty for other forms.
secs() {
    local d=0 t="$1"
    [[ "${t}" == *-* ]] && { d="${t%%-*}"; t="${t#*-}"; }
    [[ "${t}" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]] || return 0
    echo $(( d * 86400 + 10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]} ))
}
jt="$(secs "${JOB_TIME}")"; st="$(secs "${SERVER_TIME}")"
if [[ "${SERVER}" == debug && -n "${jt}" && -n "${st}" ]] && (( st < jt )); then
    echo "ERROR: SERVER_TIME ${SERVER_TIME} is shorter than JOB_TIME ${JOB_TIME}: the server would end first" >&2
    exit 1
fi

# SERVER=debug: the server job, server_job.sh with the watch, after the training job $1. Sets server_flags.
read -r -a site_extra <<< "${SBATCH_EXTRA}"
server_sbatch() {
    server_flags=(--parsable --nodes=1 --exclusive --no-requeue ${site_extra[@]+"${site_extra[@]}"}
        --account="${ACCOUNT}" --partition="${SERVER_PARTITION}" --time="${SERVER_TIME}" --dependency="after:$1"
        --job-name="lean-kimina-${RUN}" --output="${DIR}/logs/%j-kimina.out"
        --export="ALL,LEAN_KIMINA_DIR=${LEAN_KIMINA_DIR},ROOT=${ROOT},ENDPOINT_FILE=${ENDPOINT},MAX_REPLS=${MAX_REPLS},PORT=${PORT:-8000},HEADER_ROWS=${rows},TRAIN_JOB=$1,TRAIN_LOGS=${OUT}/slurm/$1-logs,HEAD_LIMIT=${HEAD_LIMIT:-600}"
        "${LEAN_KIMINA_DIR}/server_watch_job.sh")
}

K=env.nemo_gym.math_formal_lean.resources_servers.math_formal_lean
# Single quotes: the $(...) reach the driver command unexpanded and run at job start.
train_args=(
    "${K}.kimina_url="'$(KIMINA_ENDPOINT_FILE='"${ENDPOINT}"' bash '"${LEAN_KIMINA_DIR}"'/endpoint.sh url)'
    "${K}.kimina_api_key="'$(KIMINA_ENDPOINT_FILE='"${ENDPOINT}"' bash '"${LEAN_KIMINA_DIR}"'/endpoint.sh key)'
    "${K}.kimina_max_inflight=${MAX_REPLS}"
)
export RUN JOB_TIME PARTITION="${PARTITION:-normal}"
if [[ "${SERVER}" == in-job ]]; then
    # Save 30 min before the time limit (given before the overrides, so one there wins).
    if [[ -n "${jt}" ]] && (( jt > 3600 )); then
        m=$(( jt - 1800 ))
        train_args+=("checkpointing.checkpoint_must_save_by=$(printf '%02d:%02d:%02d:%02d' \
            $((m / 86400)) $((m % 86400 / 3600)) $((m % 3600 / 60)) $((m % 60)))")
    fi
    # in_job.sbatch's settings reach the job through sbatch's environment, under KIMINA_ names (ray.sub reads
    # PORT as Ray's own port).
    export BATCH_SCRIPT="${LEAN_KIMINA_DIR}/in_job.sbatch" KIMINA_DIR="${LEAN_KIMINA_DIR}" KIMINA_RL_INFRA="${ROOT}" \
        KIMINA_ENDPOINT_FILE="${ENDPOINT}" KIMINA_HEADER_ROWS="${rows}" KIMINA_REPLS="${MAX_REPLS}" \
        KIMINA_HEAD_LIMIT="${HEAD_LIMIT:-600}"
else
    # --no-requeue after launch.sh's --requeue (added for preemptable) wins: a requeued run would restart against
    # a server that is gone.
    export SBATCH_EXTRA="${SBATCH_EXTRA:+${SBATCH_EXTRA} }--no-requeue"
fi
train_args+=("$@")
# run.sh runs from the repository root below; a relative RECIPE is meant from where train.sh was started.
[[ -z "${RECIPE:-}" ]] || export RECIPE="$(realpath "${RECIPE}")"

cd "${ROOT}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    DRY_RUN=1 bash run.sh envs/lean-kimina-sft.yaml "${train_args[@]}"
    [[ "${SERVER}" == in-job ]] && exit 0
    echo
    echo "== the server job (submitted after the training, which is <training job> here):"
    server_sbatch '<training job>'
    printf 'sbatch'; printf ' %q' "${server_flags[@]}"; echo
    exit 0
fi

mkdir -p "${DIR}/logs"
out="$(bash run.sh envs/lean-kimina-sft.yaml "${train_args[@]}" | tee /dev/stderr)"
train_job="$(grep -oE "^Run .* is job [0-9]+\.$" <<< "${out}" | grep -oE '[0-9]+\.$' | tr -d .)" || true
[[ -n "${train_job}" ]] || { echo "ERROR: no training job ID in run.sh's output; no server submitted" >&2; exit 1; }

if [[ "${SERVER}" == in-job ]]; then
    cat << INFO

Training ${train_job} (${PARTITION}, ${JOB_TIME}) with its Kimina server inside the job.
  After it starts, check:
    ${OUT}/slurm/${train_job}-logs/STARTED_RAY_HEAD     (about 1 min after the start)
    ${OUT}/slurm/${train_job}-${RUN}.out                ([kimina ...] lines)
    ${DIR}/logs/${train_job}-r0-kimina-0.out            (server up, ${MAX_REPLS} REPLs, controls passed)
    ${OUT}/slurm/${train_job}-logs/ray-driver.log       ("kimina-endpoint: ... answered", then the steps)
INFO
    exit 0
fi

server_sbatch "${train_job}"
if ! server_job="$(sbatch "${server_flags[@]}")"; then
    echo "ERROR: the server job was not submitted; cancelling training ${train_job}, which would have no verifier" >&2
    scancel "${train_job}"
    exit 1
fi
cat << INFO

Training ${train_job} (${PARTITION}) and its Kimina server ${server_job} (${SERVER_PARTITION}, starts after it).
  After the training starts, check:
    ${OUT}/slurm/${train_job}-logs/STARTED_RAY_HEAD     (about 1 min after the start)
    ${DIR}/logs/${server_job}-kimina.out                ([watch ...] lines)
    ${OUT}/slurm/${train_job}-logs/ray-driver.log       ("kimina-endpoint: ... answered", then the steps)
INFO
