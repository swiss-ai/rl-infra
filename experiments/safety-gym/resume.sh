#!/usr/bin/env bash
set -euo pipefail

# Edit settings here. Adaptive is the selected run; set SAFETY_PROFILE=legacy
# explicitly to use the historical single-turn launcher. Keep RUN/OUT fixed on resume.
USER="${USER:-$(id -un)}"
SCRATCH="${SCRATCH:-/scratch/${USER}}"
SAFETY_PROFILE="${SAFETY_PROFILE:-adaptive}"
SAFETY_RESUME="${SAFETY_RESUME:-1}"
if [[ "${SAFETY_PROFILE}" == legacy ]]; then
  RUN="${RUN:-msafetemp-chonk-16k}"
else
  SAFETY_COALLOCATE="${SAFETY_COALLOCATE:-1}"
  if [[ "$SAFETY_COALLOCATE" == 1 ]]; then
    RUN="${RUN:-msafetemp-adaptive-stage1-74h-coalloc-retry2}"
  else
    RUN="${RUN:-msafetemp-adaptive-stage1-72h-retry1}"
  fi
  SAFETY_ADAPTIVE_ROOT="${SAFETY_ADAPTIVE_ROOT:-/scratch/${USER}/safetygym-adaptive}"
  ADAPTIVE_GYM_COMMIT="${ADAPTIVE_GYM_COMMIT:-1508cb2aa7000225c3d47fc49b0d67246ccc7c90}"
  OUT="${OUT:-${SAFETY_ADAPTIVE_ROOT}/runs}"
  SAFETY_ADAPTIVE_USER_OUT="${OUT}"
  SAFETY_DATA_SOURCE="${SAFETY_DATA_SOURCE:-${SAFETY_ADAPTIVE_ROOT}/data/route-e1e8a681}"
  WANDB_KEY_FILE="${WANDB_KEY_FILE:-${SCRATCH}/safetygym/runs/private/wandb.key}"
fi
JUDGE_KIND="${JUDGE_KIND:-oss}"
JUDGE_URL="${JUDGE_URL:-http://node1878:8080/v1}"
PARTITION="intTest"
if [[ "${SAFETY_PROFILE}" != legacy && "${SAFETY_COALLOCATE:-0}" == 1 ]]; then
  JOB_TIME="74:00:00"
  CHECKPOINT_DEADLINE="03:00:00:00"
  SAFETY_ADAPTIVE_JUDGE_WAIT_SECONDS=3600
else
  JOB_TIME="72:00:00"
  CHECKPOINT_DEADLINE="02:23:30:00"
fi
VALIDATION_SAMPLES=500
OUT="${OUT:-${SCRATCH}/safetygym/runs}"
SAFETY_CHONK_SOURCE="/appsdata/hanint05/hf-exports/bridge/chonk-lc-16k-v0.2-iter_0018315"
SAFETY_CHONK_MODEL="${SAFETY_CHONK_MODEL:-${SCRATCH}/safetygym/models/chonk-lc-16k-v0.2-iter_0018315-safetygym}"
TRAIN_IMAGE="${SCRATCH}/images/nemo-rl-apertus2-cfcfede4-ucclfix.sqsh"
SAFETY_DATA_SOURCE="${SAFETY_DATA_SOURCE:-${SCRATCH}/datasets/msafetemp-safetygym}"
WANDB_PROJECT="safety-gym"
WANDB_ENTITY="nutemmax-epfl"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-${OUT}/private/wandb.key}"

if (( $# != 0 )); then
  echo "This launcher takes no arguments; edit its settings at the top." >&2
  exit 2
fi
export USER SCRATCH RUN JUDGE_KIND JUDGE_URL PARTITION JOB_TIME CHECKPOINT_DEADLINE SAFETY_COALLOCATE SAFETY_ADAPTIVE_JUDGE_WAIT_SECONDS
export SAFETY_PROFILE SAFETY_RESUME ADAPTIVE_GYM_COMMIT SAFETY_ADAPTIVE_ROOT SAFETY_ADAPTIVE_USER_OUT
export VALIDATION_SAMPLES OUT SAFETY_CHONK_SOURCE SAFETY_CHONK_MODEL TRAIN_IMAGE SAFETY_DATA_SOURCE WANDB_PROJECT WANDB_ENTITY WANDB_KEY_FILE
if [[ -n "${WANDB_PROJECT}" && -z "${WANDB_API_KEY:-}" && ! -r "${WANDB_KEY_FILE}" ]]; then
  echo "W&B is enabled; provide a readable key at WANDB_KEY_FILE or set WANDB_API_KEY." >&2
  exit 2
fi
SLURM_USER="${USER}"
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
if [[ "${SAFETY_PROFILE}" != legacy && "${SAFETY_COALLOCATE:-0}" == 1 ]]; then
  coalloc_mode=--submit
  [[ "${SAFETY_COALLOCATE_TEST_ONLY:-0}" != 1 ]] || coalloc_mode=--test-only
  exec bash "${HERE}/coallocate.sh" "$coalloc_mode" "${RUN}"
fi
bash "${HERE}/run.sh" --check "${RUN}"
exec bash "${HERE}/run.sh" --submit "${RUN}"
