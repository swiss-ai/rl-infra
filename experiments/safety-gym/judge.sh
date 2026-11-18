#!/usr/bin/env bash
# One separately managed judge through rl-infra's exact one-node container launcher.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

sml_judge() {
  local action="$1" user scratch root model_dir model_id served_name sml_bin sml_root environment render_dir endpoint_file workdir framework_args sml_config run_id
  user="${USER:-$(id -un)}"
  scratch="${SCRATCH:-/scratch/${user}}"
  root="${SAFETY_ADAPTIVE_ROOT:-${scratch}/safetygym-adaptive}"
  model_dir="${SAFETY_JUDGE_DIR:-${scratch}/models/gpt-oss-safeguard-120b}"
  served_name="openai/gpt-oss-safeguard-120b"
  model_id="${user}/${served_name}"
  sml_root="${SAFETY_MODEL_LAUNCH_ROOT:-${root}/cache/model-launch}"
  sml_bin="${SML_BIN:-${root}/cache/sml-model-launch-venv/bin/sml}"
  environment="${SAFETY_SML_ENVIRONMENT:-${sml_root}/src/swiss_ai_model_launch/assets/envs/vllm_kisti.toml}"
  render_dir="${SAFETY_SML_RENDER_DIR:-${root}/runtime/sml-judge-render}"
  endpoint_file="${SAFETY_ADAPTIVE_JUDGE_ENDPOINT_FILE:-}"
  run_id="${RUN:-}"
  workdir="${SML_WORKING_DIR:-${scratch}/sml}"
  sml_config="${SML_CONFIG_DIR:-${root}/runtime/sml-config}"
  export SML_CONFIG_DIR="$sml_config"
  [[ -x "$sml_bin" ]] || { echo "Pinned SML CLI not found: $sml_bin" >&2; return 2; }
  [[ -s "$environment" ]] || { echo "KISTI SML environment not found: $environment" >&2; return 2; }
  [[ -s "${model_dir}/config.json" ]] || { echo "Judge model config missing: ${model_dir}/config.json" >&2; return 2; }
  [[ -s "${model_dir}/model.safetensors.index.json" || -n "$(find "$model_dir" -maxdepth 1 -type f -name '*.safetensors' -print -quit)" ]] || {
    echo "Judge model weights are missing under $model_dir" >&2; return 2;
  }
  if [[ ! -s "${sml_config}/config.yml" ]]; then
    mkdir -p "$sml_config"
    cat > "${sml_config}/config.yml" <<'YAML'
name: init_config
type: chain
chain:
  - name: launcher_configuration
    type: branch
    head_configuration:
      name: launcher
      type: options
      options:
        firecrest: [FirecREST, "not used by the local Slurm launch"]
        slurm: [SLURM, "local KISTI jobs"]
      value: slurm
    branches:
      firecrest: null
      slurm: null
  - name: swissai_research_api_key
    type: password
    value: local-only-disabled-opentela
YAML
    chmod 600 "${sml_config}/config.yml"
  fi
  framework_args="--model ${model_dir} --host 0.0.0.0 --max-model-len 32768 --gpu-memory-utilization 0.90 --tensor-parallel-size 1"
  local -a args=(advanced --partition "${SAFETY_KISTI_PARTITION:-intTest}" --account "${SML_ACCOUNT:-in0138}" \
    --framework vllm --environment "$environment" --container-runtime pyxis \
    --sbatch-arg='--comment=field=nlp;appl=vllm' --sbatch-arg=--gres=gpu:4 \
    --sbatch-arg=--propagate=NONE \
    --served-model-name "$served_name" --replicas 1 --nodes-per-replica 1 \
    --time "${SAFETY_JUDGE_TIME:-72:00:00}" --max-job-time "${SAFETY_JUDGE_MAX_JOB_TIME:-72:00:00}" \
    --disable-opentela --disable-metrics --pre-launch-cmds "pip uninstall -y -q torchcodec || true" \
    --framework-args "$framework_args" --no-tui)
  mkdir -p "$render_dir" "$workdir/logs"
  "$sml_bin" "${args[@]}" --output-script "$render_dir"
  [[ -s "${render_dir}/master.sh" ]] || { echo "SML did not render ${render_dir}/master.sh" >&2; return 1; }
  python3.12 "${HERE}/../../../archive/safetygym/render_judge_kisti.py" "${render_dir}/master.sh"
  python3.12 - "${render_dir}/master.sh" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
for old, new in (("#SBATCH --output=logs/%j/log.out", "#SBATCH --output=logs/%j.log"), ("#SBATCH --error=logs/%j/log.err", "#SBATCH --error=logs/%j.err")):
    if text.count(old) != 1:
        raise SystemExit(f"Refusing unexpected SML log directive: {old}")
    text = text.replace(old, new)
if "curl " in text or "https://" in text:
    raise SystemExit("External endpoint remains in sanitized local SML render")
path.write_text(text, encoding="utf-8")
PY
  bash -n "${render_dir}/master.sh"
  if [[ "$action" == render ]]; then
    printf 'Rendered SML judge script: %s/master.sh\nModel ID: %s\nTime: %s (single job; no consecutive chain)\n' \
      "$render_dir" "$model_id" "${SAFETY_JUDGE_TIME:-72:00:00}"
    return
  fi

  if [[ -z "$endpoint_file" && -n "$run_id" ]]; then
    endpoint_file="${root}/runtime/judge-endpoint-${run_id}.json"
  fi
  [[ -n "$endpoint_file" ]] || { echo 'Set SAFETY_ADAPTIVE_JUDGE_ENDPOINT_FILE (or RUN) for per-run endpoint discovery.' >&2; return 2; }
  [[ ! -e "$endpoint_file" ]] || { echo "Refusing to overwrite existing judge endpoint metadata: $endpoint_file" >&2; return 2; }
  mkdir -p "$(dirname "$endpoint_file")"
  local sbatch_bin submit_output job_id train_job_id
  sbatch_bin="${SBATCH_BIN:-$(command -v sbatch || true)}"
  [[ -n "$sbatch_bin" && -x "$sbatch_bin" ]] || { echo 'sbatch not found; rendered SML script is ready.' >&2; return 2; }
  train_job_id="${SAFETY_JUDGE_AFTER_JOB_ID:-}"
  local -a sbatch_args=(--parsable --chdir="$workdir")
  if [[ -n "$train_job_id" ]]; then
    [[ "$train_job_id" =~ ^[0-9]+$ ]] || { echo 'SAFETY_JUDGE_AFTER_JOB_ID must be a Slurm job ID.' >&2; return 2; }
    sbatch_args+=("--dependency=after:${train_job_id}")
  fi
  submit_output="$(cd "$workdir" && "$sbatch_bin" "${sbatch_args[@]}" "${render_dir}/master.sh" 2>&1)" || {
    printf '%s\n' "$submit_output" >&2; return 1;
  }
  job_id="${submit_output%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch --parsable output: $submit_output" >&2; return 1; }
  printf 'SML judge job submitted: %s\nServed model: %s\n' "$job_id" "$model_id"
  local timeout start state node api_models
  timeout="${SAFETY_JUDGE_READY_TIMEOUT_SECONDS:-900}"
  start="$(date +%s)"
  while (( $(date +%s) - start < timeout )); do
    local status_line
    status_line="$(squeue --jobs "$job_id" --noheader --format='%T %N' 2>/dev/null | head -n 1 || true)"
    state="${status_line%% *}"
    node="${status_line#* }"
    if [[ "$state" =~ ^(FAILED|CANCELLED|TIMEOUT|COMPLETED|NODE_FAIL|OUT_OF_MEMORY)$ ]]; then
      echo "Judge Slurm job ${job_id} entered ${state} before readiness." >&2; return 1;
    fi
    if [[ "$state" == RUNNING && "$node" != "$status_line" && "$node" != "(null)" && "$node" != "n/a" ]]; then
      node="${node%%,*}"
      if api_models="$(srun --jobid "$job_id" --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 --nodelist="$node" \
        curl --fail --silent --show-error --max-time 5 "http://${node}:8080/v1/models" 2>/dev/null)" && \
        python3.12 -c 'import json,sys; expected=sys.argv[1]; data=json.load(sys.stdin); raise SystemExit(0 if any(x.get("id")==expected for x in data.get("data",[])) else 1)' "$model_id" <<<"$api_models"; then
        python3.12 - "$endpoint_file" "$job_id" "$node" "$model_id" "$run_id" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
path, job_id, host, model, run_id = sys.argv[1:]
record = {
    "job_id": job_id,
    "host": host,
    "base_url": f"http://{host}:8080/v1",
    "model": model,
    "expected_walltime_seconds": 259200,
    "readiness_verified_utc": datetime.now(timezone.utc).isoformat(),
}
if run_id:
    record["run_id"] = run_id
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".judge-endpoint-")
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True)
        stream.write("\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
except BaseException:
    try: os.unlink(tmp)
    except FileNotFoundError: pass
    raise
PY
        printf 'Ready: %s (job %s); endpoint metadata: %s\n' "http://${node}:8080/v1" "$job_id" "$endpoint_file"
        return
      fi
    fi
    sleep 10
  done
  echo "Timed out after ${timeout}s waiting for judge job ${job_id} and model ${model_id}; job remains submitted." >&2
  return 1
}

if [[ "${1:-}" == --sml-render || "${1:-}" == --sml-submit ]]; then
  action=render; [[ "$1" == --sml-submit ]] && action=submit
  shift
  (($# == 0)) || { echo 'Usage: judge.sh --sml-render|--sml-submit' >&2; exit 2; }
  sml_judge "$action"
  exit $?
fi
mode="${1:---dry-run}"; (($# == 0)) || shift
export JUDGE_KIND="${1:-oss}"
[[ "$mode" == --dry-run || "$mode" == --submit ]] || { echo 'Usage: judge.sh --dry-run|--submit oss|nemotron' >&2; exit 2; }
source "${HERE}/environment.sh"
export IMAGE_WORK="${JUDGE_IMAGE:-/appsdata/${USER}/sml/images/vllm-v0.32.0-cu129-arm64.sqsh}"
export IMAGE_MASTER="${IMAGE_WORK}"
export CHECK_SCRIPT=../../../experiments/safety-gym/judge_server.sh CHECK_CPUS=288
export DRY_RUN=1
if [[ "$mode" == --dry-run ]]; then
  export OUT="${TMPDIR:-/tmp}/safetygym-${USER}-judge-${JUDGE_KIND}"
else
  export DRY_RUN=0
  [[ -s "${SAFETY_JUDGE_DIR}/config.json" && -s "${IMAGE_WORK}" ]] || { echo 'Judge model and owned JUDGE_IMAGE must exist.' >&2; exit 1; }
fi
runtime="${OUT}/runtime/judge-${JUDGE_KIND}-$(date +%Y%m%d-%H%M%S)-$$"
export SAFETY_JUDGE_CONFIG="${runtime}/judge.yaml"
# A judge uses the existing working SML vLLM image, with no training overlay mounts.
python3.12 - "$runtime" <<'PY'
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1]); p.mkdir(parents=True, exist_ok=True)
mounts = [os.environ['HOME'], os.environ['SCRATCH'], '/appsdata/' + os.environ['USER']]
mounts.append(os.environ['SAFETY_EXPERIMENT_DIR'] + ':/opt/nemo-rl')
text = 'image = "@@IMAGE_WORK@@"\nwritable = true\nmounts = [\n'
text += ''.join('  ' + json.dumps(mount) + ',\n' for mount in mounts) + ']\n[env]\n'
for key in ('SAFETY_JUDGE_MODEL', 'SAFETY_JUDGE_DIR', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    text += key + ' = ' + json.dumps(os.environ.get(key, '1')) + '\n'
(p / 'container.toml').write_text(text)
PY
export CONTAINER_SPEC="${runtime}/container.toml"
export NEMO_RL_DIR MEGATRON_DIR BRIDGE_DIR VLLM_RUNNER VLLM_APERTUS2 PIN_FILE
export PYDEPS_APERTUS2="${PYDEPS_APERTUS2:-}"
echo "One judge job, one node / four allocated GPUs, one GPU used: ${JUDGE_KIND}. No training submitted."
exec bash "${APERTUS2_DIR}/check_env_job.sh"
