#!/bin/bash
# Run one script of this folder as a one-node Slurm job inside the container (sft.sh submits train.sh with it):
#   bash backends/posttraining/submit.sh <run name> <script> [args...]
#   e.g.  bash backends/posttraining/submit.sh v11-check check.sh
# Account, partition, mandatory sbatch flags, mounts and the container runtime come from the cluster profile
# (sites/<SITE>.env). JOB_TIME (default 01:30:00), PARTITION (default the site's DEFAULT_PARTITION) and DEPENDENCY
# (e.g. afterok:<job>) go to sbatch; RESUME=1 makes the job requeueable; DRY_RUN=1 prints the command instead.
# The job log is $SFT_ROOT/logs/<jobid>-<run name>.out.
set -euo pipefail

PT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${PT_DIR}/../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"   # the container image is the Apertus 2 NeMo-RL one
source "${ROOT}/sites/load.sh" && rl_infra_load "${PT_DIR}/env" "${APERTUS2_DIR}/env"
[[ $# -ge 2 ]] || { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 1; }
NAME=$1; SCRIPT=$2; shift 2
[[ -f "${PT_DIR}/${SCRIPT}" ]] || { echo "ERROR: ${PT_DIR}/${SCRIPT} not found" >&2; exit 1; }
[[ -d "${POSTTRAINING_DIR}/src/post_training" ]] \
    || { echo "ERROR: no posttraining clone at ${POSTTRAINING_DIR}; run backends/posttraining/setup.sh" >&2; exit 1; }

# Environment of the job inside the container; it wins over values the image sets itself. The site's settings
# (SITE_ENV, and SINGLE_NODE_ENV, as this is a one-node job) come first.
ENV_VARS=()
for kv in ${SITE_ENV[@]+"${SITE_ENV[@]}"} ${SINGLE_NODE_ENV[@]+"${SINGLE_NODE_ENV[@]}"}; do
    export "${kv?}"; ENV_VARS+=("${kv%%=*}")
done
export PT_DIR POSTTRAINING_DIR SFT_ROOT SFT_PYDEPS MODELS_DIR HF_MIXTURE_DIR WANDB_KEY_FILE
export PYTHONPATH="${SFT_PYDEPS}:${POSTTRAINING_DIR}/src"
export HF_HOME="${SCRATCH}/huggingface" HF_DATASETS_CACHE="${SFT_ROOT}/hf_datasets_cache"
export TRITON_CACHE_DIR=/tmp/triton-sft TORCH_EXTENSIONS_DIR="${SFT_ROOT}/torch_extensions"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false UV_CACHE_DIR="${SFT_ROOT}/uv-cache"
# The Apertus 2 image points uv at CSCS's PyPI mirror, which HANGANG cannot reach.
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.org/simple}"
# One node: the four GPUs talk over NVLink, and NCCL needs no network plugin (on HANGANG it fails to start, as
# Slurm gives a one-node job no Slingshot service; on Alps the CXI hook is off below).
export NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1
ENV_VARS+=(PT_DIR POSTTRAINING_DIR SFT_ROOT SFT_PYDEPS MODELS_DIR HF_MIXTURE_DIR WANDB_KEY_FILE PYTHONPATH HF_HOME
           HF_DATASETS_CACHE TRITON_CACHE_DIR TORCH_EXTENSIONS_DIR PYTHONNOUSERSITE TOKENIZERS_PARALLELISM UV_CACHE_DIR
           UV_DEFAULT_INDEX NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE)

mkdir -p "${SFT_ROOT}/logs"
read -r -a extra <<< "${SBATCH_EXTRA:-}"
# A whole node (on Alps, preemptable otherwise gives the job one CPU and no GPU).
flags=(--nodes=1 --exclusive --account="${ACCOUNT}" --partition="${PARTITION:-${DEFAULT_PARTITION}}"
       --time="${JOB_TIME:-01:30:00}" --job-name="sft-${NAME}" --output="${SFT_ROOT}/logs/%j-${NAME}.out"
       ${extra[@]+"${extra[@]}"})
[[ -n "${EXCLUDE_NODES:-}" ]] && flags+=(--exclude="${EXCLUDE_NODES}")
[[ -n "${DEPENDENCY:-}" ]] && flags+=(--dependency="${DEPENDENCY}")
# A run that resumes from its checkpoints may be requeued, for example after preemption.
[[ "${RESUME:-0}" == 1 ]] && flags+=(--requeue)

# One task with all CPUs of the node, host memory on the CPU NUMA nodes 0-3 (each GH200's HBM is a NUMA node of its
# own). --overlap: the batch step holds the node, and a step asking for all its CPUs would otherwise wait for it.
printf -v args ' %q' "$@"
wrap="srun --ntasks=1 --cpus-per-task=${CPUS_PER_NODE} --cpu-bind=none --mem-bind=mask_mem:0xf --overlap"
case "${CONTAINER_RUNTIME}" in
pyxis)
    mounts="$(for m in "${SITE_MOUNTS[@]}"; do printf '%s:%s,' "${m}" "${m}"; done)"
    wrap+=" --container-image=${IMAGE_WORK} --container-mounts=${mounts%,} --no-container-mount-home"
    wrap+=" --container-writable --container-workdir=${POSTTRAINING_DIR} --container-env=$(IFS=,; echo "${ENV_VARS[*]}")"
    ;;
edf)
    # The Container Engine reads the container from an environment definition file (EDF), written here for each
    # job. The image brings its own libfabric and NCCL plugin, hence no CXI hook.
    mkdir -p "${SFT_ROOT}/edf"
    EDF="${SFT_ROOT}/edf/${NAME}.toml"
    {
        printf 'image = "%s"\n' "${IMAGE_WORK}"
        printf 'mounts = [%s]\n' "$(printf '"%s", ' "${SITE_MOUNTS[@]}" | sed 's/, $//')"
        printf 'writable = true\nworkdir = "%s"\n\n' "${POSTTRAINING_DIR}"
        printf '[annotations]\ncom.hooks.cxi.enabled = "false"\ncom.hooks.netstack.source = "host"\n\n[env]\n'
        printf 'PATH = "%s"\n' "${SFT_CONTAINER_PATH}"
        for var in "${ENV_VARS[@]}"; do printf '%s = "%s"\n' "${var}" "${!var}"; done
    } > "${EDF}"
    wrap+=" --environment=${EDF}"
    ;;
*)
    echo "ERROR: CONTAINER_RUNTIME must be pyxis or edf (is '${CONTAINER_RUNTIME}')" >&2; exit 1 ;;
esac
wrap+=" bash ${PT_DIR}/${SCRIPT}${args}"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'sbatch'; printf ' %q' "${flags[@]}"; printf ' --wrap %q\n' "${wrap}"; exit 0
fi
cd "${SFT_ROOT}/logs"   # HANGANG: jobs are submitted from scratch
sbatch "${flags[@]}" --wrap "${wrap}"
echo "  job log: ${SFT_ROOT}/logs/<jobid>-${NAME}.out"
