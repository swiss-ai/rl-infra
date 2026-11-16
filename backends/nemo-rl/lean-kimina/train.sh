#!/bin/bash
# Train on the Lean task (envs/lean-kimina-sft.yaml) with a Kimina server of its own: submits the training job
# through run.sh, then the server job with --dependency=after:<training>, and prints both job IDs.
#   DRY_RUN=1 bash backends/nemo-rl/lean-kimina/train.sh [overrides...]   setup, preflight and both sbatch commands
#   bash backends/nemo-rl/lean-kimina/train.sh [overrides...]             submits both
# Overrides go to run.sh as usual (e.g. grpo.max_num_steps=3). See RUNBOOK.md for a first run.
#
# Why this order: a server job submitted first may start hours before the training, or end before it. Here the
# server waits for the training to start, on debug (it starts 1-node jobs within about a minute), and the
# training reads the server's address and key at job start through endpoint.sh, which waits for a live server.
# The server job (server_watch_job.sh) stops when the training is over, and cancels a training whose Ray head is
# not up after HEAD_LIMIT seconds or whose server died.
#
# Settings (environment):
#   RECIPE           run.sh's recipe; default the task's own. For an SFT checkpoint, a recipe that names its
#                    sft_export, e.g. recipes/apertus2-chonk-sftv02-18315-vllm-adamw-code.yaml.
#   RUN              run name; default lean-kimina-<date>-<time>. The server's endpoint and log go to
#                    $SCRATCH/lean-kimina/runs/<RUN>/.
#   PARTITION        training partition, default normal (never preempted). preemptable starts sooner but was
#                    preempted after about 22 min; the run is then over (no requeue: its server would be gone).
#   JOB_TIME         training time limit, default 01:20:00.
#   SERVER_PARTITION default debug; SERVER_TIME default 01:30:00 (debug's maximum). It must cover JOB_TIME, as
#                    the server starts at the earliest with the training.
#   HEAD_LIMIT       seconds the server job gives the training's Ray head, default 600.
#   MAX_REPLS        the server's REPLs, default 128 (about 2 GB each).
# Debug runs one job per user: while another debug job of yours runs, the server waits, and the training waits
# for it in endpoint.sh for at most 30 min (KIMINA_ENDPOINT_WAIT), then fails.
set -euo pipefail
LEAN_KIMINA_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${LEAN_KIMINA_DIR}/../../.." && pwd)"
# sbatch passes the whole environment into both jobs, and ray.sub prints most of it: keep secrets out. W&B reads
# its key from WANDB_KEY_FILE (see site.env.example).
unset WANDB_API_KEY HF_TOKEN OPENAI_API_KEY
source "${ROOT}/sites/load.sh" && rl_infra_load "${LEAN_KIMINA_DIR}/env"
source "${LEAN_KIMINA_DIR}/pin"
source "${LEAN_KIMINA_DIR}/kimina.sh"
[[ -x "$(kimina_repl_bin)" ]] || { echo "ERROR: no Kimina install in ${KIMINA_INSTALL}; run setup.sh first" >&2; exit 1; }
rows="${HEADER_ROWS:-${LEAN_DATA}/data/train.v434.jsonl}"
[[ -s "${rows}" ]] || { echo "ERROR: no training rows ${rows}; run setup-data.sh first" >&2; exit 1; }

RUN="${RUN:-lean-kimina-$(date +%Y%m%d-%H%M%S)}"
JOB_TIME="${JOB_TIME:-01:20:00}"
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
if [[ -n "${jt}" && -n "${st}" ]] && (( st < jt )); then
    echo "ERROR: SERVER_TIME ${SERVER_TIME} is shorter than JOB_TIME ${JOB_TIME}: the server would end first" >&2
    exit 1
fi

# The server job: server_job.sh with the watch, after the training job $1. Sets server_flags.
read -r -a site_extra <<< "${SBATCH_EXTRA}"
server_sbatch() {
    server_flags=(--parsable --nodes=1 --exclusive --no-requeue ${site_extra[@]+"${site_extra[@]}"}
        --account="${ACCOUNT}" --partition="${SERVER_PARTITION}" --time="${SERVER_TIME}" --dependency="after:$1"
        --job-name="lean-kimina-${RUN}" --output="${DIR}/logs/%j-kimina.out"
        --export="ALL,LEAN_KIMINA_DIR=${LEAN_KIMINA_DIR},ROOT=${ROOT},ENDPOINT_FILE=${ENDPOINT},MAX_REPLS=${MAX_REPLS:-128},PORT=${PORT:-8000},HEADER_ROWS=${rows},TRAIN_JOB=$1,TRAIN_LOGS=${OUT}/slurm/$1-logs,HEAD_LIMIT=${HEAD_LIMIT:-600}"
        "${LEAN_KIMINA_DIR}/server_watch_job.sh")
}

K=env.nemo_gym.math_formal_lean.resources_servers.math_formal_lean
# Single quotes: the $(...) reach the driver command unexpanded and run at job start.
train_args=(
    "${K}.kimina_url="'$(KIMINA_ENDPOINT_FILE='"${ENDPOINT}"' bash '"${LEAN_KIMINA_DIR}"'/endpoint.sh url)'
    "${K}.kimina_api_key="'$(KIMINA_ENDPOINT_FILE='"${ENDPOINT}"' bash '"${LEAN_KIMINA_DIR}"'/endpoint.sh key)'
    "${K}.kimina_max_inflight=${MAX_REPLS:-128}"
    "$@"
)
# --no-requeue after launch.sh's --requeue (added for preemptable) wins: a requeued run would restart against a
# server that is gone.
export RUN JOB_TIME PARTITION="${PARTITION:-normal}" SBATCH_EXTRA="${SBATCH_EXTRA:+${SBATCH_EXTRA} }--no-requeue"
# run.sh runs from the repository root below; a relative RECIPE is meant from where train.sh was started.
[[ -z "${RECIPE:-}" ]] || export RECIPE="$(realpath "${RECIPE}")"

cd "${ROOT}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    DRY_RUN=1 bash run.sh envs/lean-kimina-sft.yaml "${train_args[@]}"
    echo
    echo "== the server job (submitted after the training, which is <training job> here):"
    server_sbatch '<training job>'
    printf 'sbatch'; printf ' %q' "${server_flags[@]}"; echo
    exit 0
fi

out="$(bash run.sh envs/lean-kimina-sft.yaml "${train_args[@]}" | tee /dev/stderr)"
train_job="$(grep -oE "^Run .* is job [0-9]+\.$" <<< "${out}" | grep -oE '[0-9]+\.$' | tr -d .)" || true
[[ -n "${train_job}" ]] || { echo "ERROR: no training job ID in run.sh's output; no server submitted" >&2; exit 1; }

mkdir -p "${DIR}/logs"
server_sbatch "${train_job}"
if ! server_job="$(sbatch "${server_flags[@]}")"; then
    echo "ERROR: the server job was not submitted; cancelling training ${train_job}, which would have no verifier" >&2
    scancel "${train_job}"
    exit 1
fi
cat << EOF

Training ${train_job} (${PARTITION}) and its Kimina server ${server_job} (${SERVER_PARTITION}, starts after it).
  After the training starts, check:
    ${OUT}/slurm/${train_job}-logs/STARTED_RAY_HEAD     (about 1 min after the start)
    ${DIR}/logs/${server_job}-kimina.out                ([watch ...] lines)
    ${OUT}/slurm/${train_job}-logs/ray-driver.log       ("kimina-endpoint: ... answered", then the steps)
EOF
