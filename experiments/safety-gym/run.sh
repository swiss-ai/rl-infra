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
  [[ -s "${SAFETY_DATA_SOURCE}/manifest.json" && -s "${SAFETY_DATA_SOURCE}/adaptive_train.jsonl" && -s "${SAFETY_ADAPTIVE_VALIDATION_FILE}" && -s "${SAFETY_ADAPTIVE_VALIDATION_FILE}.manifest.json" ]] || {
    echo 'Adaptive immutable data/500-row validation view is missing its manifest or JSONL files.' >&2; exit 1; }
  [[ -s "${SAFETY_ADAPTIVE_ROOT}/native-runtime-manifest.json" ]] || { echo 'Feature native runtime manifest is missing.' >&2; exit 1; }
  [[ -d "${SAFETY_GYM_SITE_PACKAGES}" ]] || { echo 'Prepared Gym site-packages for the NRL policy interpreter are missing.' >&2; exit 1; }
  [[ -s "${SAFETY_GYM_DIR}/responses_api_agents/adaptive_safety/app.py" ]] || { echo 'Feature Gym export is missing the adaptive agent.' >&2; exit 1; }
  for path in "${VLLM_RUNNER}" "${VLLM_APERTUS2}"; do
    [[ -s "$path" ]] || { echo "Pinned vLLM bind source is missing: $path" >&2; exit 1; }
  done
  source "${APERTUS2_DIR}/pin"
  [[ "$(git hash-object "${VLLM_RUNNER}")" == "${VLLM_RUNNER_BLOB}" ]] || {
    echo 'Staged vLLM runner does not match the pinned image override.' >&2; exit 1; }
  case "${SAFETY_RESUME:-0}" in 0|1) ;; *) echo 'SAFETY_RESUME must be 0 or 1.' >&2; exit 2 ;; esac
  adaptive_run_dir="${OUT}/results/${RUN}"
  adaptive_run_manifest="${adaptive_run_dir}/adaptive-objective-manifest.json"
  if [[ -e "${adaptive_run_dir}" ]]; then
    [[ "${SAFETY_RESUME:-0}" == 1 ]] || {
      echo 'Adaptive run already exists; choose a fresh RUN/OUT for a new objective or set SAFETY_RESUME=1.' >&2; exit 2; }
    [[ -s "${adaptive_run_manifest}" ]] || { echo 'Cannot resume adaptive run without its original objective manifest.' >&2; exit 2; }
    compgen -G "${adaptive_run_dir}/step_*" >/dev/null || { echo 'Cannot resume adaptive run without a native checkpoint.' >&2; exit 2; }
  else
    [[ "${SAFETY_RESUME:-0}" == 0 ]] || { echo 'SAFETY_RESUME=1 requires the existing adaptive run directory.' >&2; exit 2; }
  fi
  if [[ -e "${SAFETY_ADAPTIVE_JUDGE_ENDPOINT_FILE}" && "${SAFETY_RESUME:-0}" != 1 && "${IN_ALLOCATION:-0}" != 1 ]]; then
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
if [[ "${SAFETY_PROFILE}" != legacy && "$mode" != --dry-run ]]; then
  manifest_inputs=(
    --input "adaptive_data_manifest=${SAFETY_DATA_SOURCE}/manifest.json"
    --input "adaptive_train=${SAFETY_DATA_SOURCE}/adaptive_train.jsonl"
    --input "adaptive_validation=${SAFETY_ADAPTIVE_VALIDATION_FILE}"
    --input "adaptive_validation_manifest=${SAFETY_ADAPTIVE_VALIDATION_FILE}.manifest.json"
    --input "native_runtime_manifest=${SAFETY_ADAPTIVE_ROOT}/native-runtime-manifest.json"
    --input "adaptive_config=${HERE}/adaptive.yaml"
    --input "adaptive_limits=${HERE}/adaptive_limits.yaml"
    --input "legacy_base_config=${HERE}/chonk.yaml"
    --input "model_alias_config=${HERE}/adaptive_models.yaml"
    --input "gym_agent_config=${SAFETY_ADAPTIVE_GYM_CONFIG}"
    --input "gym_agent_app=${SAFETY_GYM_DIR}/responses_api_agents/adaptive_safety/app.py"
    --input "gym_agent_policy=${SAFETY_GYM_DIR}/responses_api_agents/adaptive_safety/policy.py"
    --input "gym_agent_prompts=${SAFETY_GYM_DIR}/responses_api_agents/adaptive_safety/prompts.yaml"
    --input "gym_project=${SAFETY_GYM_DIR}/pyproject.toml"
    --input "gym_dependency_lock=${SAFETY_GYM_DIR}/uv.lock"
    --input "runtime_bridge=${HERE}/adaptive_bridge.py"
    --input "runtime_overlay=${HERE}/adaptive_overlay.py"
    --input "runtime_native_adapter=${HERE}/adaptive_runtime_overlay.py"
    --input "runtime_compat=${HERE}/compat.py"
    --input "runtime_renderer=${HERE}/render.py"
    --input "runtime_driver=${HERE}/driver.sh"
    --input "runtime_train_entrypoint=${HERE}/train.py"
    --input "runtime_storage_logging=${HERE}/storage_logging.py"
    --input "runtime_run_launcher=${HERE}/run.sh"
    --input "runtime_environment=${HERE}/environment.sh"
    --input "runtime_container_template=${APERTUS2_DIR}/edf.toml.template"
    --input "runtime_python_dependencies=${APERTUS2_DIR}/pydeps.sh"
    --input "model_config=${SAFETY_CHONK_MODEL}/config.json"
    --input "model_weights_index=${SAFETY_CHONK_MODEL}/model.safetensors.index.json"
    --input "model_tokenizer_config=${SAFETY_CHONK_MODEL}/tokenizer_config.json"
    --input "model_tokenizer=${SAFETY_CHONK_MODEL}/tokenizer.json"
  )
  pinned_json="$(python3.12 - "$RUN" "${SAFETY_PROFILE}" <<'PY'
import json, os, sys
print(json.dumps({
    "run_id": sys.argv[1],
    "profile": sys.argv[2],
    "gym_commit": os.environ.get("SAFETY_GYM_COMMIT", "unknown"),
    "policy_model": os.environ.get("SAFETY_CHONK_SOURCE", ""),
    "policy_model_overlay": os.environ.get("SAFETY_CHONK_MODEL", ""),
    "attacker_model": os.environ.get("SAFETY_SWISSAI_ATTACKER_MODEL", ""),
    "attacker_revision": os.environ.get("SAFETY_SWISSAI_MODEL_REVISION", "unknown"),
    "judge_model": os.environ.get("SAFETY_ADAPTIVE_JUDGE_MODEL", ""),
    "judge_revision": os.environ.get("SAFETY_ADAPTIVE_JUDGE_REVISION", "unknown"),
    "validator_model": os.environ.get("SAFETY_SWISSAI_VALIDATOR_MODEL", ""),
    "validator_revision": os.environ.get("SAFETY_SWISSAI_MODEL_REVISION", "unknown"),
    "native_image": os.environ.get("IMAGE_WORK", ""),
    "ordinal_curriculum": "attested-native-prompt-ordinal-or-frozen-stage-id",
    "source_contract_schema": "safety-source-v1",
    "validation_view": os.environ.get("SAFETY_ADAPTIVE_VALIDATION_FILE", ""),
    "validation_samples": os.environ.get("SAFETY_VALIDATION_SAMPLES", "500"),
}, sort_keys=True))
PY
)"
  manifest_args=(manifest --output "${adaptive_run_manifest}" "${manifest_inputs[@]}" --pinned-json "${pinned_json}")
  if [[ "${SAFETY_RESUME:-0}" == 1 ]]; then manifest_args+=(--resume); fi
  if [[ "$mode" == --submit || "${SAFETY_RESUME:-0}" == 1 ]]; then
    python3.12 "${HERE}/adaptive_bridge.py" "${manifest_args[@]}"
  fi
fi
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
if [[ "${SAFETY_PROFILE}" != legacy && "${mode}" == --submit ]]; then
  if [[ "${IN_ALLOCATION:-0}" == 1 ]]; then
    [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo 'IN_ALLOCATION requires a Slurm job ID.' >&2; exit 2; }
    mkdir -p "$(dirname "${SAFETY_ADAPTIVE_TRAINING_JOB_FILE}")"
    python3.12 - "${SAFETY_ADAPTIVE_TRAINING_JOB_FILE}" "${RUN}" "${SLURM_JOB_ID}" "${SAFETY_RESUME:-0}" <<'PY'
import json, os, sys, tempfile
from pathlib import Path
path, run_id, job_id, resume = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
if path.exists() and resume != "1":
    raise SystemExit(f"Refusing to overwrite training job metadata: {path}")
fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "training_job_id": job_id}, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temp_path, 0o600)
    os.replace(temp_path, path)
finally:
    if os.path.exists(temp_path):
        os.unlink(temp_path)
PY
    IN_ALLOCATION=1 bash "${APERTUS2_DIR}/launch.sh" "${train_config}" 12 "$RUN" \
      "checkpointing.checkpoint_must_save_by=${SAFETY_CHECKPOINT_DEADLINE}" "$@"
    exit $?
  fi
  mkdir -p "$(dirname "${SAFETY_ADAPTIVE_TRAINING_JOB_FILE}")"
  rm -f "${SAFETY_ADAPTIVE_TRAINING_JOB_FILE}"
  submitted="$(bash "${APERTUS2_DIR}/launch.sh" "${train_config}" 12 "$RUN" \
    "checkpointing.checkpoint_must_save_by=${SAFETY_CHECKPOINT_DEADLINE}" "$@")"
  printf '%s\n' "${submitted}"
  python3.12 - "${SAFETY_ADAPTIVE_TRAINING_JOB_FILE}" "${RUN}" "${submitted}" <<'PY'
import json
import os
from pathlib import Path
import re
import sys
import tempfile

path, run_id, output = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
matches = re.findall(r"^Submitted batch job ([0-9]+)$", output, re.MULTILINE)
if not matches:
    raise SystemExit("Adaptive submit did not return a Slurm job ID; judge handoff sidecar was not written.")
job_id = matches[-1]
fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "training_job_id": job_id}, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temp_path, 0o600)
    os.replace(temp_path, path)
finally:
    if os.path.exists(temp_path):
        os.unlink(temp_path)
PY
else
  exec bash "${APERTUS2_DIR}/launch.sh" "${train_config}" 12 "$RUN" \
    "checkpointing.checkpoint_must_save_by=${SAFETY_CHECKPOINT_DEADLINE}" "$@"
fi
