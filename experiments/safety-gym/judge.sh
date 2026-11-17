#!/usr/bin/env bash
# One separately managed judge through rl-infra's exact one-node container launcher.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
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
