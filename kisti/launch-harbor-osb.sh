#!/bin/bash
# Chonk GRPO on Harbor tasks with OpenSandbox sandboxes, on KISTI HANGANG: the KISTI counterpart of
# run-harbor-osb.sh + backends/nemo-rl/apertus2/launch.sh. Same recipe, task file, image and driver; what
# changes is the container runtime (pyxis instead of the CSCS EDF) and the site settings.
#
#   source kisti/env.sh && bash kisti/launch-harbor-osb.sh [overrides...]
#
# Needs on KISTI (see docs/harbor-osb-chonk-run.md, section KISTI HANGANG):
#   $KISTI_IMAGE, $KISTI_MODEL                        David's image and the Chonk checkpoint
#   $KISTI_SCRATCH/projects/{nemo-rl-apertus2,megatron-lm-moe-apertus2,megatron-bridge-apertus2}
#                                                     the checkouts setup.sh + setup-gym.sh build on Alps
#   $HARBOR_ROOT (default $KISTI_SCRATCH/osb/harbor_micro)  tasks/, train.jsonl, validation.jsonl
#   ~/.opensandbox_token                              OpenSandbox API key, one line (chmod 600)
# RECIPE (default: the 12-node blueprint), RUN, JOB_TIME, OPENSANDBOX_DOMAIN, OVERLAY override the defaults.
set -euo pipefail

RL_INFRA="$(cd "$(dirname "$0")/.." && pwd)"
source "${RL_INFRA}/kisti/env.sh"
TASK="${RL_INFRA}/envs/harbor-osb.yaml"
RECIPE="$(realpath "${RECIPE:-${RL_INFRA}/recipes/apertus2-chonk-vllm-adamw.yaml}")"
NODES="${NODES:-$(sed -n 's/^  num_nodes: *\([0-9]*\).*/\1/p' "${RECIPE}" | head -1)}"
RUN="${RUN:-kisti-harbor-osb-$(date +%m%d-%H%M)}"
JOB_TIME="${JOB_TIME:-04:00:00}"

PROJECTS="${KISTI_SCRATCH}/projects"
NEMO_RL_DIR="${NEMO_RL_DIR:-${PROJECTS}/nemo-rl-apertus2}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECTS}/megatron-lm-moe-apertus2}"
BRIDGE_DIR="${BRIDGE_DIR:-${PROJECTS}/megatron-bridge-apertus2}"
OUT="${OUT:-${KISTI_SCRATCH}/nemo-rl}"
for f in "${KISTI_IMAGE}" "${KISTI_MODEL}/config.json" "${NEMO_RL_DIR}/ray.sub" "${MEGATRON_DIR}" "${BRIDGE_DIR}/src" \
         "${NEMO_RL_DIR}/3rdparty/Gym-workspace/Gym/nemo_gym/__init__.py"; do
    [[ -e "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done

# --- OpenSandbox (as run-harbor-osb.sh) -----------------------------------------------------------
if [[ -z "${OPENSANDBOX_API_KEY:-}" ]]; then
    [[ -r "${HOME}/.opensandbox_token" ]] || { echo "ERROR: no OPENSANDBOX_API_KEY and no ~/.opensandbox_token" >&2; exit 1; }
    OPENSANDBOX_API_KEY="$(tr -d '\n' < "${HOME}/.opensandbox_token")"
fi
export OPENSANDBOX_API_KEY OPENSANDBOX_DOMAIN="${OPENSANDBOX_DOMAIN:-https://opensandbox-gvisor.breithorn.svc.cscs.ch}"
code="$(curl -s -m 15 -o /dev/null -w '%{http_code}' -H "OPEN-SANDBOX-API-KEY: ${OPENSANDBOX_API_KEY}" \
    "${OPENSANDBOX_DOMAIN%/}/v1/sandboxes?pageSize=1" || true)"
[[ "${code}" == 200 ]] || { echo "ERROR: OpenSandbox rejected the key or is unreachable (HTTP ${code:-none})" >&2; exit 1; }
export RL_REGISTRY_USERNAME="${RL_REGISTRY_USERNAME:-anonymous}" RL_REGISTRY_PASSWORD="${RL_REGISTRY_PASSWORD:-unused}"
export NEMO_GYM_TEAM="${NEMO_GYM_TEAM:-infra01}" NEMO_GYM_USER="${NEMO_GYM_USER:-${USER}}"
export NEMO_GYM_WORKLOAD="${NEMO_GYM_WORKLOAD:-chonk-harbor-osb-kisti}" NEMO_GYM_RUN_ID="${NEMO_GYM_RUN_ID:-${RUN}}"

# --- Harbor tasks and Gym venvs -------------------------------------------------------------------
export HARBOR_ROOT="${HARBOR_ROOT:-${KISTI_SCRATCH}/osb/harbor_micro}"
export HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:-${HARBOR_ROOT}/tasks}"
for f in "${HARBOR_ROOT}/train.jsonl" "${HARBOR_ROOT}/validation.jsonl" "${HARBOR_TASKS_DIR}"; do
    [[ -e "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done
export NEMO_GYM_VENV_DIR="${NEMO_GYM_VENV_DIR:-${HARBOR_ROOT}/venvs}" UV_LINK_MODE=copy
mkdir -p "${HARBOR_ROOT}/jobs" "${NEMO_GYM_VENV_DIR}" "${OUT}/slurm" "${OUT}/results" "${OUT}/logs" "${OUT}/configs"

# --- The EDF's [env], as exports (pyxis passes the job environment into the container) -------------
export SCRATCH="${KISTI_SCRATCH}"   # the recipes read ${oc.env:SCRATCH}/models/apertus2-chonk-lc-iter3576
PYDEPS="${KISTI_SCRATCH}/nemo-rl-apertus2/pydeps"
export PYDEPS PATH="/usr/local/lib/python3.12/dist-packages/torch_tensorrt/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="/opt/nemo-rl:/opt/nemo-rl/3rdparty/Automodel-workspace/Automodel:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:${BRIDGE_DIR}/src:${MEGATRON_DIR}:${PYDEPS}:${PYDEPS_KISTI}"
export PMIX_MCA_psec=native NCCL_PXN_DISABLE=1 NCCL_CROSS_NIC=0 UCCL_EP_TRANSPORT=cxi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false RAY_USAGE_STATS_ENABLED=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 VLLM_WORKER_MULTIPROC_METHOD=spawn HF_DEACTIVATE_ASYNC_LOAD=1 NRL_DISABLE_NUMA_BINDING=1
export HF_HOME="${SCRATCH}/huggingface/" VLLM_CACHE_ROOT="${SCRATCH}/vllm_cache" FLASHINFER_WORKSPACE_BASE="${SCRATCH}/.cache/flashinfer"
export XDG_CACHE_HOME="${SCRATCH}/.cache" UV_CACHE_DIR="${SCRATCH}/.cache/uv" TORCH_EXTENSIONS_DIR="${SCRATCH}/.cache/torch_extensions"
export TRITON_CACHE_DIR=/tmp/triton-nemo-rl-apertus2 TORCHINDUCTOR_CACHE_DIR=/tmp/nemo_rl_jit/torchinductor MEGATRON_CONFIG_LOCK_DIR=/tmp/megatron-config-locks

# --- KISTI site settings ------------------------------------------------------------------------------
# The image's uv points at CSCS's JFrog mirror, which KISTI cannot reach.
export UV_DEFAULT_INDEX=https://pypi.org/simple
# KISTI: LL128 gives wrong inter-node all-reduce results on HANGANG.
export NCCL_PROTO='^LL128' FI_PROVIDER=cxi NCCL_IB_DISABLE=1
# The image sets FI_CXI_RDZV_PROTO=alt_read, which wants this Slurm network option on every step.
export SLURM_NETWORK=disable_rdzv_get

# --- Config, driver command, ray.sub -------------------------------------------------------------------
CFG="${OUT}/configs/${RUN}.yaml"
printf 'defaults:\n  - %s\n  - %s\n' "${RECIPE}" "${TASK}" > "${CFG}"
# OVERLAY: optional task overlay merged last, e.g. envs/harbor-osb-proxy-sft.yaml
[[ -z "${OVERLAY:-}" ]] || printf '  - %s\n' "$(realpath "${OVERLAY}")" >> "${CFG}"
COMMAND="cd /opt/nemo-rl && \
bash ${RL_INFRA}/kisti/pydeps-kisti.sh && \
RAY_DEDUP_LOGS=0 DRIVER_RETRIES=0 bash ${RL_INFRA}/envs/reap_after.sh bash ${RL_INFRA}/backends/nemo-rl/driver_retry.sh \
bash ${RL_INFRA}/backends/nemo-rl/apertus2/driver.sh examples/nemo_gym/run_grpo_nemo_gym.py --config ${CFG} \
cluster.num_nodes=${NODES} cluster.gpus_per_node=4 \
checkpointing.checkpoint_dir=${OUT}/results/${RUN} logger.log_dir=${OUT}/logs/${RUN} $*"

# ray.sub's pyxis path, plus a writable container (vLLM and FlashInfer write caches under the image's paths)
# and --container-env: for variables the image itself sets (UV_DEFAULT_INDEX is one), the image's value wins
# over the job's unless pyxis is told to take the job's.
RAY_SUB="${OUT}/slurm/ray-kisti.sub"
sed 's|COMMON_SRUN_ARGS+=" --container-image=$CONTAINER"|COMMON_SRUN_ARGS+=" --container-image=$CONTAINER --container-writable --container-env=UV_DEFAULT_INDEX"|' \
    "${NEMO_RL_DIR}/ray.sub" > "${RAY_SUB}"
grep -q -- '--container-writable' "${RAY_SUB}" || { echo "ERROR: could not patch ray.sub for a writable container" >&2; exit 1; }

MOUNTS="${NEMO_RL_DIR}:/opt/nemo-rl,${MEGATRON_DIR}:${MEGATRON_DIR},${BRIDGE_DIR}:${BRIDGE_DIR},${KISTI_SCRATCH}:${KISTI_SCRATCH},${RL_INFRA}:${RL_INFRA}"
echo "== ${RUN}: ${NODES} nodes on ${KISTI_PARTITION}, OpenSandbox ${OPENSANDBOX_DOMAIN}, tasks ${HARBOR_TASKS_DIR}"
cd "${OUT}/slurm"
CONTAINER="${KISTI_IMAGE}" MOUNTS="${MOUNTS}" COMMAND="${COMMAND}" GPUS_PER_NODE=4 RAY_SINGLE_SRUN=1 BASE_LOG_DIR="${OUT}/slurm" \
SRUN_MEM_BIND="${SRUN_MEM_BIND-verbose,mask_mem:0xf}" NUMA_WATCH="${NUMA_WATCH-${RL_INFRA}/backends/nemo-rl/numa_watch.sh}" \
    sbatch "${KISTI_SLURM[@]}" --nodes="${NODES}" --time="${JOB_TIME}" --job-name="nrl-${RUN}" \
    --output="${OUT}/slurm/%j-${RUN}.out" "${RAY_SUB}"
