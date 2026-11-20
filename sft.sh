#!/bin/bash
# SFT of an Apertus-v1.1 base model on one node of four GPUs, with swiss-ai/posttraining (backends/posttraining):
#   bash sft.sh <size: 0.5B | 1.5B | 4B> <conversations> [Hydra overrides...]
#   e.g.  bash sft.sh 1.5B 75000       about one hour          bash sft.sh 1.5B 320000      about four hours
# The conversations are a random subset of the Apertus SFT mixture (seed 42). Every launch first brings the
# posttraining clone up to date (login node; seconds after the first time); the job builds the Python overlay and
# converts the mixture on first use, then trains. The final model is linked as $SFT_ROOT/models/<run name>
# ($SCRATCH/apertus-v11-sft/models/...), the folder GRPO takes as policy.model_name. Recipe, run sizes and results:
# docs/apertus-v11.md.
#
#   bash sft.sh ...             full run: logs to W&B (refuses to start without the key file)
#   DEBUG=1 bash sft.sh ...     debug run: the name ends in -DEBUG, 3 optimizer steps, 1-hour limit
#   DRY_RUN=1 bash sft.sh ...   shows the sbatch command; no setup, no job
#
# Environment: RUN_NAME (default apertus-v11-<size>-sft-n<conversations>); LR, DEVICE_BS, GC (per-size defaults in
# backends/posttraining/train.sh); JOB_TIME (default 05:00:00), PARTITION, DEPENDENCY; RESUME=1 SAVE_STEPS=<n> keeps
# checkpoints a requeued job continues from. W&B: WANDB_PROJECT (default apertus-v11), WANDB_ENTITY, WANDB_KEY_FILE.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ $# -ge 2 ]] || { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 1; }
PT_DIR="${ROOT}/backends/posttraining"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"   # the container image is the Apertus 2 NeMo-RL one
source "${ROOT}/sites/load.sh" && rl_infra_load "${PT_DIR}/env" "${APERTUS2_DIR}/env"
SIZE=$1; N=$2

export RUN_NAME="${RUN_NAME:-apertus-v11-${SIZE}-sft-n${N}}"
extra=()
if [[ "${DEBUG:-0}" == 1 ]]; then
    [[ "${RUN_NAME}" == *-DEBUG ]] || RUN_NAME+=-DEBUG
    extra=(++training_args.max_steps=3)
    JOB_TIME="${JOB_TIME:-01:00:00}"
elif [[ ! -r "${WANDB_KEY_FILE}" ]]; then
    echo "ERROR: a full run logs to W&B: put the API key in ${WANDB_KEY_FILE} (WANDB_KEY_FILE), or DEBUG=1" >&2; exit 1
fi
export JOB_TIME="${JOB_TIME:-05:00:00}" WANDB_PROJECT="${WANDB_PROJECT:-apertus-v11}"

if [[ "${DRY_RUN:-0}" != 1 ]]; then
    LOG="${SFT_ROOT}/setup.log"; mkdir -p "${SFT_ROOT}"
    echo "== setup (log: ${LOG})"
    bash "${PT_DIR}/setup.sh" > "${LOG}" 2>&1 || { tail -n 20 "${LOG}"; echo "ERROR: setup failed, see ${LOG}" >&2; exit 1; }
fi
echo "Run:     ${RUN_NAME} ($([[ "${DEBUG:-0}" == 1 ]] && echo debug || echo full) run, W&B project ${WANDB_PROJECT}, time limit ${JOB_TIME})"
echo "Model:   ${MODELS_DIR}/Apertus-v1.1-${SIZE}, ${N} conversations"
echo "Output:  ${SFT_ROOT}/models/${RUN_NAME}"
exec bash "${PT_DIR}/submit.sh" "${RUN_NAME}" train.sh "$@" ${extra[@]+"${extra[@]}"}
