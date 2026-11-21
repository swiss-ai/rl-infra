#!/usr/bin/env bash
set -euo pipefail

# Edit settings here. Keep RUN, OUT, dataset source, and validation size fixed
# across resumes. The judge endpoint must remain alive through the allocation.
RUN="msafetemp-chonk-16k-b25"
JUDGE_KIND="oss"
JUDGE_URL="http://node1878:8080/v1"
PARTITION="intTest"
JOB_TIME="72:00:00"
CHECKPOINT_DEADLINE="02:23:30:00"
VALIDATION_SAMPLES=500
OUT="/scratch/hanint14/safetygym/runs"
SAFETY_CHONK_MODEL="/scratch/hanint14/models/chonk-lc-16k-v0.2-iter_0018315-ifgym"
TRAIN_IMAGE="/scratch/hanint14/images/nemo-rl-apertus2-cfcfede4-ucclfix.sqsh"
# Balanced view from build_balanced_data.py: 25% benign in every prompt-length bin, short direct prompts,
# benign rows tagged response_policy_mapped=refuse_then_redirect, validation with or_val_* agents.
SAFETY_DATA_SOURCE="/scratch/hanint14/datasets/msafetemp-safetygym-balanced-b25"
SAFETY_DATA_PREBUILT=1
# Staged Gym copy that contains the refuse_then_redirect policy (same commit and venv as the first run).
SAFETY_GYM_DIR="/scratch/hanint14/safetygym/Gym-6cdaeea52b25e2a48a2477ba2f8fffcf7113b2ee-orfix-py312"
# One epoch of the balanced view at 128 prompts per step.
MAX_STEPS=715
WANDB_PROJECT="safety-gym"
WANDB_ENTITY="nutemmax-epfl"
WANDB_KEY_FILE="/scratch/hanint14/safetygym/runs/private/wandb.key"

if (( $# != 0 )); then
  echo "This launcher takes no arguments; edit its settings at the top." >&2
  exit 2
fi
export RUN JUDGE_KIND JUDGE_URL PARTITION JOB_TIME CHECKPOINT_DEADLINE
export SAFETY_DATA_PREBUILT SAFETY_GYM_DIR
export VALIDATION_SAMPLES OUT SAFETY_CHONK_MODEL TRAIN_IMAGE SAFETY_DATA_SOURCE WANDB_PROJECT WANDB_ENTITY WANDB_KEY_FILE
if [[ -n "${WANDB_PROJECT}" && -z "${WANDB_API_KEY:-}" && ! -r "${WANDB_KEY_FILE}" ]]; then
  echo "W&B is enabled; provide a readable key at WANDB_KEY_FILE or set WANDB_API_KEY." >&2
  exit 2
fi
SLURM_USER="${USER:-hanint14}"
if ! active="$(timeout 10s squeue -h -u "${SLURM_USER}" -n "nrl-${RUN}" -o "%i %T")"; then
  echo "Could not check Slurm for ${RUN}; refusing to submit." >&2
  exit 1
fi
if [[ -n "${active}" ]]; then
  echo "${RUN} already has an active Slurm job; not submitting another:" >&2
  printf '%s\n' "${active}" >&2
  exit 0
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
bash "${HERE}/run.sh" --check "${RUN}" "grpo.max_num_steps=${MAX_STEPS}"
exec bash "${HERE}/run.sh" --submit "${RUN}" "grpo.max_num_steps=${MAX_STEPS}"
