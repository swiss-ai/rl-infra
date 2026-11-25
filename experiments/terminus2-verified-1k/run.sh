#!/bin/bash
# Terminus-2 on the Harbor verified tasks at scale, on Alps: envs/harbor-verified-sft.yaml (Gym entrypoint,
# Terminus-2 through harbor_agent, one OpenSandbox pod per trial plus a verifier pod) with
#   1024 sandboxes in flight   (harbor_agent concurrency; the task file has 200)
#   128 prompts x 8 = 1024     rollouts per step and policy.train_global_batch_size 1024 (one optimizer step)
#   max_trajectory_age_steps 4 (the task file has 2)
# and the run logged to Weights & Biases. README.md next to this file has the details and the risks.
#   bash experiments/terminus2-verified-1k/run.sh [overrides...]     DRY_RUN=1: the job without submitting it
# Env (defaults in brackets):
#   SANDBOXES [1024], PROMPTS [128] (x 8 rollouts; the batch follows), AGE [4], STEPS [the task file's 20],
#   RUN [t2-verified-1k-<date>], JOB_TIME [12:00:00], PARTITION [normal: Daint has no preemptable queue],
#   WANDB_PROJECT [grpo-dev], WANDB_ENTITY [apertus], the key from WANDB_API_KEY (environment) or, when that is
#   unset, from WANDB_KEY_FILE [~/.w_and_b_key; read at job start by driver.sh], WANDB=0 turns W&B off,
#   HARBOR_ROOT [$SCRATCH/osb/harbor_verified], VERIFIED_STORE [the capstor task store], COLLECT_ARGS,
#   EXCLUDE_CAMPAIGNS / EXCLUDE_TASKS [the campaign and the 5 tasks left out on Alps, see below; empty = none],
#   RECIPE [recipes/apertus2-chonk-sft19374-alps-vllm-adamw-tools.yaml, 12 nodes],
#   APERTUS_COMMON_URL [SSH], NEMO_RL_DIR / GYM_VENV_DIR / PYDEPS_GYM_DIR / EDF_NAME [this experiment's own].
# Harbor trial dirs: $SCRATCH/harbor-jobs/<RUN>.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"

# This branch pins Gym (15 patches) and NeMo-RL (without 0057) differently from chonk-sc-harbor025, whose
# runs use the default clone ${PROJECTS_DIR}/nemo-rl-apertus2. setup.sh / setup-gym.sh would reset that clone
# under its running and queued jobs (a job mounts the clone named in its EDF when it starts, not when it is
# submitted), so this experiment gets a clone, Gym venvs, a pydeps overlay and an EDF of its own.
PROJECTS_DIR="$(source "${ROOT}/sites/load.sh" > /dev/null && rl_infra_load > /dev/null && echo "${PROJECTS_DIR}")"
export NEMO_RL_DIR="${NEMO_RL_DIR:-${PROJECTS_DIR}/nemo-rl-apertus2-t2v1k}"
export GYM_VENV_DIR="${GYM_VENV_DIR:-${SCRATCH}/nemo-rl-apertus2-t2v1k/gym-venvs}"
export PYDEPS_GYM_DIR="${PYDEPS_GYM_DIR:-${SCRATCH}/nemo-rl-apertus2-t2v1k/pydeps-gym}"
export EDF_NAME="${EDF_NAME:-nemo-rl-apertus2-t2v1k}"

TASK_STORE="${VERIFIED_STORE:-/capstor/store/cscs/swissai/infra01/datasets/swe_and_agentic_gym_datasets/harbor_verified_tasks}"
HARBOR_ROOT="${HARBOR_ROOT:-${SCRATCH}/osb/harbor_verified}"
# absolute paths: HOME inside the container is not yours; the files themselves are never read here
CRED="$(realpath -m "${RL_REGISTRY_CRED_FILE:-${HOME}/.rl_registry_cred}")"
KEY="$(realpath -m "${OPENSANDBOX_API_KEY_FILE:-${HOME}/.opensandbox_token}")"
[[ -r "${KEY}" ]] || { echo "ERROR: no OpenSandbox key file ${KEY}" >&2; exit 1; }
[[ -r "${CRED}" ]] || { echo "ERROR: no registry credentials file ${CRED}" >&2; exit 1; }

# Weights & Biases (backends/nemo-rl/launch.sh): WANDB_PROJECT turns it on. The key: WANDB_API_KEY from the
# environment (sbatch passes it on to the job), else driver.sh reads WANDB_KEY_FILE at job start.
if [[ "${WANDB:-1}" != 0 ]]; then
    export WANDB_PROJECT="${WANDB_PROJECT:-grpo-dev}" WANDB_ENTITY="${WANDB_ENTITY:-apertus}"
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        export WANDB_KEY_FILE="$(realpath -m "${WANDB_KEY_FILE:-${HOME}/.w_and_b_key}")"
        [[ -r "${WANDB_KEY_FILE}" ]] || {
            echo "ERROR: W&B needs WANDB_API_KEY in the environment or the key file ${WANDB_KEY_FILE} (WANDB=0: no W&B)" >&2
            exit 1; }
    fi
else
    unset WANDB_PROJECT WANDB_ENTITY
fi

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

# Left out of training on Alps, as in the Terminus-2 runs there (a step waits for its slowest group):
# campaign swe-rebench-v2.1-rl (its pods took up to 1492 s to start, 10 of 24 hit the 1800 s start timeout), and
# 5 tasks whose verifier failed in most of their trials (all at the 1200 s cap). Rows look like
# {"instance_id": "verified::<campaign>__<task>", ...}.
EXCLUDE_CAMPAIGNS="${EXCLUDE_CAMPAIGNS-swe-rebench-v2\.1-rl}"
EXCLUDE_TASKS="${EXCLUDE_TASKS-gui-cs__terminal-gui-4860|ml-explore__mlx-3604|floci-io__floci-1932|modin-project__modin-(6369|7193)}"
TRAIN="${HARBOR_ROOT}/train.jsonl"
DROP=()
[[ -z "${EXCLUDE_CAMPAIGNS}" ]] || DROP+=(-e "\"instance_id\": \"verified::[^\"]*(${EXCLUDE_CAMPAIGNS})__")
[[ -z "${EXCLUDE_TASKS}" ]] || DROP+=(-e "\"instance_id\": \"verified::[^\"]*__(${EXCLUDE_TASKS})\"")
if (( ${#DROP[@]} )); then
    TRAIN="${HARBOR_ROOT}/train-filtered-t2v1k.jsonl"
    grep -vE "${DROP[@]}" "${HARBOR_ROOT}/train.jsonl" > "${TRAIN}.tmp.$$" || true
    [[ -s "${TRAIN}.tmp.$$" ]] || {
        echo "ERROR: EXCLUDE_CAMPAIGNS='${EXCLUDE_CAMPAIGNS}' EXCLUDE_TASKS='${EXCLUDE_TASKS}' leave no training rows" >&2
        rm -f "${TRAIN}.tmp.$$"; exit 1; }
    mv -f "${TRAIN}.tmp.$$" "${TRAIN}"
    echo "== left out $(( $(wc -l < "${HARBOR_ROOT}/train.jsonl") - $(wc -l < "${TRAIN}") )) rows" \
        "(campaigns /${EXCLUDE_CAMPAIGNS}/, tasks /${EXCLUDE_TASKS}/); training on ${TRAIN}"
fi

SANDBOXES="${SANDBOXES:-1024}" PROMPTS="${PROMPTS:-128}" AGE="${AGE:-4}"
STEP_ARGS=()
[[ -z "${STEPS:-}" ]] || STEP_ARGS=(grpo.max_num_steps="${STEPS}")
# exported before use: in `RUN=x bash run.sh ... $RUN` the argument's $RUN is expanded by the calling shell first
export RUN="${RUN:-t2-verified-1k-$(date +%Y%m%d-%H%M%S)}"
export JOB_TIME="${JOB_TIME:-12:00:00}" PARTITION="${PARTITION:-normal}"
export APERTUS_COMMON_URL="${APERTUS_COMMON_URL:-git@github.com:swiss-ai/apertus-common.git}"
JOBS="${SCRATCH}/harbor-jobs/${RUN}"
mkdir -p "${JOBS}"
RECIPE="${RECIPE:-${ROOT}/recipes/apertus2-chonk-sft19374-alps-vllm-adamw-tools.yaml}"
echo "== run ${RUN}: ${PROMPTS} x 8 = $(( PROMPTS * 8 )) rollouts per step, at most ${SANDBOXES} sandboxes," \
     "max_trajectory_age_steps ${AGE}, steps ${STEPS:-from the task file}, W&B ${WANDB_PROJECT:-off}${WANDB_ENTITY:+ (${WANDB_ENTITY})}"
echo "== recipe ${RECIPE}; clone ${NEMO_RL_DIR}; EDF ${EDF_NAME}; Harbor trial dirs in ${JOBS}"

RECIPE="${RECIPE}" \
exec bash "${ROOT}/run.sh" "${ROOT}/envs/harbor-verified-sft.yaml" \
    harbor_data_dir="${HARBOR_ROOT}" \
    data.train.data_path="${TRAIN}" \
    registry_cred_file="${CRED}" \
    env.nemo_gym.sandbox.opensandbox.connection.api_key_file="${KEY}" \
    env.nemo_gym.sandbox.opensandbox.attribution.run="${RUN}" \
    env.nemo_gym.harbor_agent.responses_api_agents.harbor_agent.harbor_jobs_dir="${JOBS}" \
    env.nemo_gym.harbor_agent.responses_api_agents.harbor_agent.concurrency="${SANDBOXES}" \
    grpo.num_prompts_per_step="${PROMPTS}" policy.train_global_batch_size="$(( PROMPTS * 8 ))" \
    grpo.async_grpo.max_trajectory_age_steps="${AGE}" \
    ${STEP_ARGS[@]+"${STEP_ARGS[@]}"} \
    "$@"
