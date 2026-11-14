#!/bin/bash
# Run V: smoke test of Kostas' verified Harbor tasks on Alps (../../envs/harbor-verified-sft.yaml). It covers
# what the micro runs cannot: verifier pods of their own ([verifier] environment_mode = "separate", Gym patches
# 0011-0012) and task images pulled from rl-registry with the credentials file (0013, 0031). Terminus-2, Gym
# entrypoint through ../../run.sh as in experiments/runs.md "SFT, Harbor verified tasks" (HANGANG job 1254979),
# not the Single Controller. One step of 8 x 8 trials, no validation, no checkpoint.
#   bash experiments/chonk-sc/run-v.sh [overrides...]        DRY_RUN=1 shows the job without submitting it
# First use: extracts the tasks into HARBOR_ROOT (~1.4 GB, ~80k files; envs/prepare_harbor_verified_tasks.sh).
# (TASK_STORE, not STORE: Alps exports STORE and sites/alps.env builds the image paths from it.)
# Env: VERIFIED_STORE (default the capstor store HANGANG mirrors), HARBOR_ROOT ($SCRATCH/osb/harbor_verified),
# COLLECT_ARGS (default --include-unverified when the store has no tasks.csv, as on HANGANG), RUN, JOB_TIME
# (02:00:00), RL_REGISTRY_CRED_FILE (~/.rl_registry_cred), OPENSANDBOX_API_KEY_FILE (~/.opensandbox_token),
# APERTUS_COMMON_URL (SSH, as run-s.sh). Harbor trial dirs: $SCRATCH/harbor-jobs/<RUN>.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"

TASK_STORE="${VERIFIED_STORE:-/capstor/store/cscs/swissai/infra01/datasets/swe_and_agentic_gym_datasets/harbor_verified_tasks}"
HARBOR_ROOT="${HARBOR_ROOT:-${SCRATCH}/osb/harbor_verified}"
# absolute paths: HOME inside the container is not yours; the files themselves are never read here
CRED="$(realpath -m "${RL_REGISTRY_CRED_FILE:-${HOME}/.rl_registry_cred}")"
KEY="$(realpath -m "${OPENSANDBOX_API_KEY_FILE:-${HOME}/.opensandbox_token}")"
[[ -r "${CRED}" ]] || { echo "ERROR: no registry credentials file ${CRED} (username:password, chmod 600)" >&2; exit 1; }
[[ -r "${KEY}" ]] || { echo "ERROR: no OpenSandbox key file ${KEY}" >&2; exit 1; }

if [[ ! -s "${HARBOR_ROOT}/train.jsonl" ]]; then
    [[ -d "${TASK_STORE}" ]] || { echo "ERROR: no task store ${TASK_STORE}; set VERIFIED_STORE" >&2; exit 1; }
    if [[ -z "${COLLECT_ARGS+x}" ]] && ! compgen -G "${TASK_STORE}/*/tasks.csv" > /dev/null; then
        echo "note: no <campaign>/tasks.csv in ${TASK_STORE}: taking every task (--include-unverified), as on HANGANG"
        export COLLECT_ARGS=--include-unverified
    fi
    echo "== preparing the verified tasks in ${HARBOR_ROOT}"
    bash "${ROOT}/envs/prepare_harbor_verified_tasks.sh" "${TASK_STORE}" "${HARBOR_ROOT}"
fi
echo "== tasks: $(wc -l < "${HARBOR_ROOT}/train.jsonl") train, $(wc -l < "${HARBOR_ROOT}/validation.jsonl") validation rows in ${HARBOR_ROOT}"

# exported before use: in `RUN=x bash run.sh ... $RUN` the argument's $RUN is expanded by the calling shell first
export RUN="${RUN:-sft19374-harbor-verified-$(date +%Y%m%d-%H%M%S)}"
export JOB_TIME="${JOB_TIME:-02:00:00}"
export APERTUS_COMMON_URL="${APERTUS_COMMON_URL:-git@github.com:swiss-ai/apertus-common.git}"
JOBS="${SCRATCH}/harbor-jobs/${RUN}"
mkdir -p "${JOBS}"
echo "== run ${RUN}; Harbor trial dirs in ${JOBS}"

RECIPE="${ROOT}/recipes/apertus2-chonk-sft19374-alps-vllm-adamw-tools.yaml" \
exec bash "${ROOT}/run.sh" "${ROOT}/envs/harbor-verified-sft.yaml" \
    grpo.max_num_steps=1 grpo.val_at_start=false grpo.val_at_end=false grpo.val_period=0 \
    checkpointing.enabled=false \
    harbor_data_dir="${HARBOR_ROOT}" \
    registry_cred_file="${CRED}" \
    env.nemo_gym.sandbox.opensandbox.connection.api_key_file="${KEY}" \
    env.nemo_gym.harbor_agent.responses_api_agents.harbor_agent.harbor_jobs_dir="${JOBS}" \
    "$@"
