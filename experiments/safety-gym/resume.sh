#!/usr/bin/env bash
set -euo pipefail

# Edit settings here. Keep RUN, OUT, dataset source, and validation size fixed
# across resumes. The judge endpoint must remain alive through the allocation.
RUN="msafetemp-chonk-16k"
JUDGE_KIND="oss"
JUDGE_URL="http://node1878:8080/v1"
PARTITION="intTest"
JOB_TIME="72:00:00"
CHECKPOINT_DEADLINE="02:23:30:00"
VALIDATION_SAMPLES=500
OUT="/scratch/hanint14/safetygym/runs"
SAFETY_CHONK_MODEL="/scratch/hanint14/models/chonk-lc-16k-v0.2-iter_0018315-ifgym"
TRAIN_IMAGE="/scratch/hanint14/images/nemo-rl-apertus2-cfcfede4-ucclfix.sqsh"
SAFETY_DATA_SOURCE="/scratch/hanint14/datasets/msafetemp-safetygym"
WANDB_PROJECT="safety-gym"
WANDB_ENTITY="nutemmax-epfl"
WANDB_KEY_FILE="/scratch/hanint14/safetygym/runs/private/wandb.key"

if (( $# != 0 )); then
  echo "This launcher takes no arguments; edit its settings at the top." >&2
  exit 2
fi
export RUN JUDGE_KIND JUDGE_URL PARTITION JOB_TIME CHECKPOINT_DEADLINE
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
bash "${HERE}/run.sh" --check "${RUN}"
exec bash "${HERE}/run.sh" --submit "${RUN}"
