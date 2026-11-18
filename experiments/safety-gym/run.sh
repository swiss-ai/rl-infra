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
train_config="${TRAIN_CONFIG:-chonk.yaml}"
if [[ "${SAFETY_PROFILE}" == legacy ]]; then
  [[ "${train_config}" == chonk.yaml ]] || { echo 'Legacy profile only accepts chonk.yaml.' >&2; exit 2; }
  train_config="${HERE}/chonk.yaml"
else
  case "${SAFETY_PROFILE}" in
    adaptive|single_turn_bridge|honeypot_single_turn|adaptive_honeypot) ;;
    *) echo 'Unsupported adaptive feature profile.' >&2; exit 2 ;;
  esac
  [[ "${TRAIN_CONFIG:-adaptive.yaml}" == adaptive.yaml ]] || { echo 'Feature profiles require adaptive.yaml.' >&2; exit 2; }
  train_config="${HERE}/adaptive.yaml"
  [[ "${OUT}" == "${SAFETY_ADAPTIVE_ROOT}"/* ]] || { echo 'Adaptive OUT must stay below SAFETY_ADAPTIVE_ROOT.' >&2; exit 2; }
  export SAFETY_RUN_ID="${RUN}"
  profile_args=(--profile "${SAFETY_PROFILE}")
  case "${SAFETY_PROFILE}" in
    adaptive|adaptive_honeypot|single_turn_bridge) profile_args+=(--adaptive) ;;
  esac
  case "${SAFETY_PROFILE}" in
    adaptive_honeypot|honeypot_single_turn) profile_args+=(--honeypot) ;;
  esac
  [[ "${SAFETY_PROFILE}" == single_turn_bridge ]] && profile_args+=(--bridge-control)
  python3.12 "${HERE}/adaptive_bridge.py" profile "${profile_args[@]}"
  [[ -s "${SAFETY_DATA_SOURCE}/manifest.json" && -s "${SAFETY_DATA_SOURCE}/adaptive_train.jsonl" && -s "${SAFETY_DATA_SOURCE}/adaptive_validation.jsonl" ]] || {
    echo 'Adaptive immutable data view is missing its manifest or JSONL files.' >&2; exit 1; }
  [[ -s "${SAFETY_ADAPTIVE_ROOT}/native-runtime-manifest.json" ]] || { echo 'Feature native runtime manifest is missing.' >&2; exit 1; }
  [[ -s "${SAFETY_GYM_DIR}/responses_api_agents/adaptive_safety/app.py" ]] || { echo 'Feature Gym export is missing the adaptive agent.' >&2; exit 1; }
  if [[ "$mode" == --submit && -e "${OUT}/results/${RUN}" ]]; then
    echo 'Adaptive run IDs are immutable; choose a fresh RUN name instead of resuming.' >&2; exit 2
  fi
  if [[ "$mode" == --submit && -e "${SAFETY_ADAPTIVE_JUDGE_ENDPOINT_FILE}" ]]; then
    echo 'Adaptive SML endpoint file already exists; choose a fresh RUN name.' >&2; exit 2
  fi
fi
[[ "$SAFETY_VALIDATION_SAMPLES" == 500 || "$SAFETY_VALIDATION_SAMPLES" == 1000 ]] || { echo 'VALIDATION_SAMPLES must be 500 or 1000.' >&2; exit 2; }
case "${OUT}" in /scratch/"${USER}"/*|/appsdata/"${USER}"/*) ;; *) echo 'OUT must be in your scratch or appsdata.' >&2; exit 2 ;; esac
if [[ "$mode" == --dry-run ]]; then
  runtime="${TMPDIR:-/tmp}/safetygym-${USER}-${RUN}"
  export DRY_RUN=1
  # Stock dry-run prepares directories/preload. Place only those tiny artifacts in /tmp.
  real_out="$OUT"; export OUT="$runtime/preview"
else
  if [[ "${SAFETY_PROFILE}" == legacy && -z "$SAFETY_JUDGE_BASE_URL" ]]; then
    echo "Set JUDGE_URL to a running judge endpoint before launching." >&2; exit 2
  fi
  if [[ "$mode" == --check ]]; then
    runtime="${TMPDIR:-/tmp}/safetygym-${USER}-${RUN}-check"
  else
    runtime="${OUT}/runtime/${RUN}-$(date +%Y%m%d-%H%M%S)-$$"
  fi
  export DRY_RUN=0
fi
if [[ "${SAFETY_PROFILE}" == legacy ]]; then
  for path in "${SAFETY_DATA_SOURCE}/manifest.json" "${SAFETY_DATA_SOURCE}/train.jsonl" "${SAFETY_DATA_SOURCE}/validation.jsonl"; do
    [[ -s "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
  done
fi
for path in "${SAFETY_CHONK_MODEL}/config.json" "${SAFETY_CHONK_MODEL}/model.safetensors.index.json" "${MEGATRON_DIR}/megatron/core/inference/sampling_params.py"; do
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
  if [[ "${SAFETY_PROFILE}" == legacy ]]; then python3.12 "${HERE}/probe_judge.py"; fi
  NEMO_RL_DIR="$NEMO_RL_DIR" JOB_TIME="$JOB_TIME" bash "${HERE}/../../backends/nemo-rl/preflight.sh" "${train_config}" 12 "checkpointing.checkpoint_must_save_by=${SAFETY_CHECKPOINT_DEADLINE}" "$@"
  echo 'Read-only checks passed; GPU execution still requires a Slurm smoke test.'
  exit 0
fi
echo 'One training job: 12 nodes (8 trainer + 4 rollout), 48 GPUs. Judges are separate jobs.'
[[ "$mode" != --dry-run ]] || echo "Preview directories use /tmp. Actual OUT=${real_out}; no dataset or model copied."
# Overrides are passed last, exactly as the stock launcher supports.
exec bash "${APERTUS2_DIR}/launch.sh" "${train_config}" 12 "$RUN" \
  "checkpointing.checkpoint_must_save_by=${SAFETY_CHECKPOINT_DEADLINE}" "$@"
