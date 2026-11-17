#!/usr/bin/env bash
# Thin experiment wrapper around the unchanged native rl-infra launcher.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
mode="${1:---dry-run}"
case "$mode" in
  --help|-h) echo 'Usage: [JUDGE_KIND=oss|nemotron JUDGE_URL=http://node:8080/v1] bash run.sh --dry-run|--check|--submit RUN [Hydra overrides...]'; exit 0 ;;
  --dry-run|--check|--submit) shift || true ;;
  *) echo 'Expected --dry-run, --check or --submit.' >&2; exit 2 ;;
esac
RUN="${1:-safetygym-chonk}"; (($# == 0)) || shift
[[ "$RUN" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid run name.' >&2; exit 2; }
source "${HERE}/environment.sh"
[[ "$SAFETY_VALIDATION_SAMPLES" == 500 || "$SAFETY_VALIDATION_SAMPLES" == 1000 ]] || { echo 'VALIDATION_SAMPLES must be 500 or 1000.' >&2; exit 2; }
case "${OUT}" in /scratch/"${USER}"/*|/appsdata/"${USER}"/*) ;; *) echo 'OUT must be in your scratch or appsdata.' >&2; exit 2 ;; esac
if [[ "$mode" == --dry-run ]]; then
  runtime="${TMPDIR:-/tmp}/safetygym-${USER}-${RUN}"
  export DRY_RUN=1
  # Stock dry-run prepares directories/preload. Place only those tiny artifacts in /tmp.
  real_out="$OUT"; export OUT="$runtime/preview"
else
  [[ -n "$SAFETY_JUDGE_BASE_URL" ]] || { echo 'Set JUDGE_URL to a running judge endpoint before launching.' >&2; exit 2; }
  if [[ "$mode" == --check ]]; then
    runtime="${TMPDIR:-/tmp}/safetygym-${USER}-${RUN}-check"
  else
    runtime="${OUT}/runtime/${RUN}-$(date +%Y%m%d-%H%M%S)-$$"
  fi
  export DRY_RUN=0
fi
for path in "${SAFETY_DATA_SOURCE}/manifest.json" "${SAFETY_DATA_SOURCE}/train.jsonl" "${SAFETY_DATA_SOURCE}/validation.jsonl" "${SAFETY_CHONK_MODEL}/config.json" "${SAFETY_CHONK_MODEL}/model.safetensors.index.json" "${MEGATRON_DIR}/megatron/core/inference/sampling_params.py"; do
  [[ -s "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
done
case "$IMAGE_WORK" in /scratch/"${USER}"/*|/appsdata/"${USER}"/*) ;; *) echo 'TRAIN_IMAGE must belong to your scratch/appsdata tree.' >&2; exit 2 ;; esac
if [[ "$mode" != --dry-run && ! -s "$IMAGE_WORK" ]]; then
  echo "Set TRAIN_IMAGE to an existing owned compatible NeMo-RL image: $IMAGE_WORK is absent." >&2; exit 1
fi
export SAFETY_JUDGE_CONFIG="${JUDGE_CONFIG:-${runtime}/judge.yaml}"
python3.12 "${HERE}/render.py" "$runtime"
export CONTAINER_SPEC="${runtime}/container.toml"
export NEMO_RL_DIR MEGATRON_DIR BRIDGE_DIR VLLM_RUNNER VLLM_APERTUS2 PIN_FILE
export PYDEPS_APERTUS2="${PYDEPS_APERTUS2:-}"
if [[ "$mode" == --check ]]; then
  python3.12 "${HERE}/probe_judge.py"
  NEMO_RL_DIR="$NEMO_RL_DIR" JOB_TIME="$JOB_TIME" bash "${HERE}/../../backends/nemo-rl/preflight.sh" "${HERE}/chonk.yaml" 12 "checkpointing.checkpoint_must_save_by=${SAFETY_CHECKPOINT_DEADLINE}" "$@"
  echo 'Read-only checks passed; GPU execution still requires a Slurm smoke test.'
  exit 0
fi
echo 'One training job: 12 nodes (8 trainer + 4 rollout), 48 GPUs. Judges are separate jobs.'
[[ "$mode" != --dry-run ]] || echo "Preview directories use /tmp. Actual OUT=${real_out}; no dataset or model copied."
# Overrides are passed last, exactly as the stock launcher supports.
exec bash "${APERTUS2_DIR}/launch.sh" "${HERE}/chonk.yaml" 12 "$RUN" \
  "checkpointing.checkpoint_must_save_by=${SAFETY_CHECKPOINT_DEADLINE}" "$@"
