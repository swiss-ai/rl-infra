#!/bin/bash
# Train Apertus 2 Chonk with GRPO on Harbor SWE tasks whose containers run on OpenSandbox:
#   bash run-harbor-osb.sh [overrides...]
# Same recipe, launcher and image as run.sh (envs/gsm8k.yaml); this script adds what the Gym
# path needs: the OpenSandbox endpoint and key, the Harbor task tree, and Gym's venv directory
# on the shared filesystem. Run setup first (run.sh does it) so the clone, image and model exist.
#
#   ~/.opensandbox_token   OpenSandbox API key, one line, no username (chmod 600)
#   HARBOR_ROOT            train.jsonl, validation.jsonl, tasks/ (default $SCRATCH/osb/harbor_micro)
#   RECIPE, RUN, PARTITION, JOB_TIME, ACCOUNT, OPENSANDBOX_DOMAIN, OVERLAY override the defaults.
#   QOS                    optional Slurm QOS (exported as SBATCH_QOS); partition normal needs none.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TASK="${ROOT}/envs/harbor-osb.yaml"
RECIPE="$(realpath "${RECIPE:-${ROOT}/recipes/apertus2-chonk-vllm-adamw.yaml}")"
NODES="$(yq r "${RECIPE}" cluster.num_nodes)"
RUN="${RUN:-harbor-osb-$(date +%Y%m%d-%H%M%S)}"
export ACCOUNT="${ACCOUNT:-infra01}" PARTITION="${PARTITION:-normal}" JOB_TIME="${JOB_TIME:-02:00:00}"
[[ -z "${QOS:-}" ]] || export SBATCH_QOS="${QOS}"

APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${APERTUS2_DIR}/env"
source "${ROOT}/site.env"
export SCRATCH
mkdir -p "${OUT}/configs"

# --- OpenSandbox -----------------------------------------------------------------------------
# Exported before sbatch on purpose: Gym resolves ${oc.env:...} inside Ray workers on every node,
# and those inherit the job's environment, not the driver command's.
if [[ -z "${OPENSANDBOX_API_KEY:-}" ]]; then
    [[ -r "${HOME}/.opensandbox_token" ]] || { echo "ERROR: no OPENSANDBOX_API_KEY and no ~/.opensandbox_token" >&2; exit 1; }
    OPENSANDBOX_API_KEY="$(tr -d '\n' < "${HOME}/.opensandbox_token")"
fi
export OPENSANDBOX_API_KEY
export OPENSANDBOX_DOMAIN="${OPENSANDBOX_DOMAIN:-https://opensandbox-gvisor.breithorn.svc.cscs.ch}"
# /health needs no key, so it says nothing about the key. List sandboxes instead: a rejected key
# (HTTP 401) would otherwise only show up as masked rollouts, after the job waited in the queue.
code="$(curl -s -m 15 -o /dev/null -w '%{http_code}' -H "OPEN-SANDBOX-API-KEY: ${OPENSANDBOX_API_KEY}" \
    "${OPENSANDBOX_DOMAIN%/}/v1/sandboxes?pageSize=1" || true)"
[[ "${code}" == 200 ]] || { echo "ERROR: OpenSandbox rejected the key or is unreachable (HTTP ${code:-none}) at ${OPENSANDBOX_DOMAIN}" >&2; exit 1; }

# The task images are in a public Harbor project; the provider still sends image_auth.
export RL_REGISTRY_USERNAME="${RL_REGISTRY_USERNAME:-anonymous}" RL_REGISTRY_PASSWORD="${RL_REGISTRY_PASSWORD:-unused}"

# Attribution labels on every sandbox pod, and the run id the reaper filters on.
export NEMO_GYM_TEAM="${NEMO_GYM_TEAM:-infra01}" NEMO_GYM_USER="${NEMO_GYM_USER:-${USER}}"
export NEMO_GYM_WORKLOAD="${NEMO_GYM_WORKLOAD:-chonk-harbor-osb}" NEMO_GYM_RUN_ID="${NEMO_GYM_RUN_ID:-${RUN}}"

# --- Harbor tasks and Gym venvs ----------------------------------------------------------------
export HARBOR_ROOT="${HARBOR_ROOT:-${SCRATCH}/osb/harbor_micro}"
export HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:-${HARBOR_ROOT}/tasks}"
for f in "${HARBOR_ROOT}/train.jsonl" "${HARBOR_ROOT}/validation.jsonl" "${HARBOR_TASKS_DIR}"; do
    [[ -e "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done
mkdir -p "${HARBOR_ROOT}/jobs"
# Harbor dispatches each rollout as a Ray task whose runtime_env carries the absolute path of the
# harbor_agent venv, so the venvs must be on a filesystem every node sees.
export NEMO_GYM_VENV_DIR="${NEMO_GYM_VENV_DIR:-${HARBOR_ROOT}/venvs}" UV_LINK_MODE=copy
mkdir -p "${NEMO_GYM_VENV_DIR}"
[[ -f "${NEMO_RL_DIR}/3rdparty/Gym-workspace/Gym/nemo_gym/__init__.py" ]] \
    || { echo "ERROR: no Gym checkout in ${NEMO_RL_DIR}/3rdparty/Gym-workspace/Gym" >&2; exit 1; }

CFG="${OUT}/configs/${RUN}.yaml"
printf 'defaults:\n  - %s\n  - %s\n' "${RECIPE}" "${TASK}" > "${CFG}"
# OVERLAY: optional task overlay merged last, e.g. envs/harbor-osb-proxy-sft.yaml
[[ -z "${OVERLAY:-}" ]] || printf '  - %s\n' "$(realpath "${OVERLAY}")" >> "${CFG}"
echo "== ${RUN}: ${NODES} nodes, partition ${PARTITION}, OpenSandbox ${OPENSANDBOX_DOMAIN}, tasks ${HARBOR_TASKS_DIR}"
SCRIPT=examples/nemo_gym/run_grpo_nemo_gym.py \
    bash "${APERTUS2_DIR}/launch.sh" "${CFG}" "${NODES}" "${RUN}" "$@"
