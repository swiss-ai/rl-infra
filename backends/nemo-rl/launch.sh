#!/bin/bash
# Submit a NeMo-RL run: ray.sub inside the stack's container, run with the cluster profile's
# container runtime (sites/<SITE>.env: the EDF installed by setup.sh on Alps, pyxis flags on HANGANG).
#
# Usage: bash backends/nemo-rl/launch.sh <config> <num_nodes> <run_name> [overrides...]
#   <config>     a YAML in this repo (e.g. experiments/grpo-math/qwen3-32b-4n4g-fsdp2tp2.yaml),
#                any other path on a mounted filesystem, or an upstream recipe relative to
#                the NeMo-RL clone (e.g. examples/configs/grpo_math_1B.yaml)
#   overrides    passed to the training script, e.g. grpo.max_num_steps=50
#   SCRIPT=examples/run_vlm_grpo.py for vision-language models (default: examples/run_grpo.py)
#   PARTITION, JOB_TIME, ACCOUNT, USE_RESERVATION, EXCLUDE_NODES override the configuration.
#   DEPENDENCY=afterok:<jobid> starts the job only after another one succeeded.
#   PREFLIGHT=0 skips the config check (preflight.sh) done before submission.
#   PIN_FILE, DRIVER_PYTHON select another NeMo-RL source/image stack (see apertus2/launch.sh).
#   DRY_RUN=1 prints the sbatch command and the container setup instead of submitting.
#   WANDB_PROJECT (and WANDB_ENTITY), e.g. in site.env, log the run to Weights & Biases under
#   its run name; the API key is read from WANDB_KEY_FILE (default ~/.wandb_api_key) at job start.
#
# Everything a job writes (Slurm/Ray logs, checkpoints, metrics) goes under $OUT on
# scratch; this repo and the NeMo-RL clone are only read.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
source "${ROOT}/sites/load.sh" && rl_infra_load
source "${PIN_FILE}"
source "${HERE}/container.sh"

CONFIG=$1; NODES=$2; RUN=$3; shift 3
SCRIPT="${SCRIPT:-examples/run_grpo.py}"
PARTITION="${PARTITION:-${DEFAULT_PARTITION}}"
JOB_TIME="${JOB_TIME:-02:00:00}"
# No reservation by default; USE_RESERVATION=<name> to use one.
RESERVATION="${USE_RESERVATION:-}"

# The driver runs inside the container (cwd /opt/nemo-rl), so the config must be an
# absolute path it can see: the site's filesystems are mounted at the same paths, the
# NeMo-RL clone at /opt/nemo-rl.
if [[ -f "${CONFIG}" ]]; then
    CONFIG="$(realpath "${CONFIG}")"
elif [[ -f "${ROOT}/${CONFIG}" ]]; then
    CONFIG="${ROOT}/${CONFIG}"
elif [[ -f "${NEMO_RL_DIR}/${CONFIG}" ]]; then
    CONFIG="/opt/nemo-rl/${CONFIG}"
else
    echo "ERROR: config ${CONFIG} not found (cwd, ${ROOT} or ${NEMO_RL_DIR})" >&2; exit 1
fi

# Check the config here, before it waits in the queue (PREFLIGHT=0 skips it). Its report
# goes to slurm/<jobid>-<run>.preflight.txt; only errors and warnings are printed.
PREFLIGHT_OUT=""
if [[ "${PREFLIGHT:-1}" == 1 ]]; then
    if command -v yq >/dev/null; then
        PREFLIGHT_OUT="$(mktemp)"
        trap 'rm -f "${PREFLIGHT_OUT}"' EXIT
        if ! NEMO_RL_DIR="${NEMO_RL_DIR}" JOB_TIME="${JOB_TIME}" bash "${HERE}/preflight.sh" "${CONFIG}" "${NODES}" "$@" > "${PREFLIGHT_OUT}" 2>&1; then
            cat "${PREFLIGHT_OUT}" >&2
            echo "ERROR: preflight failed, not submitting (PREFLIGHT=0 to override)" >&2; exit 1
        fi
        grep "WARNING" "${PREFLIGHT_OUT}" >&2 || true
    else
        echo "WARNING: yq not found, config not checked" >&2
    fi
fi

# Re-stage the image if the scratch purge removed it. Copy to a per-process temp name,
# then rename: the real path only ever holds a complete image, and two launches that
# both find it missing each finish their own copy (the last rename wins) instead of
# interleaving writes into one file.
if [[ ! -s "${IMAGE_WORK}" && "${DRY_RUN:-0}" == 1 ]]; then
    echo "WARNING: image ${IMAGE_WORK} missing (a real launch stages it from ${IMAGE_MASTER:-<no master>})" >&2
elif [[ ! -s "${IMAGE_WORK}" ]]; then
    [[ "${IMAGE_WORK}" != "${IMAGE_MASTER}" && -s "${IMAGE_MASTER}" ]] \
        || { echo "ERROR: image ${IMAGE_WORK} missing and no master to stage it from" >&2; exit 1; }
    echo "Re-staging $(basename "${IMAGE_WORK}") from ${IMAGE_MASTER}..."
    mkdir -p "$(dirname "${IMAGE_WORK}")"
    tmp="${IMAGE_WORK}.tmp.$$"
    cp "${IMAGE_MASTER}" "${tmp}" || { rm -f "${tmp}"; echo "ERROR: image copy failed" >&2; exit 1; }
    mv "${tmp}" "${IMAGE_WORK}"
fi

# Warning only, it does not stop the submission. The image's venvs were built for the
# pinned commit; at job start NeMo-RL compares the clone's pyproject.toml, uv.lock and submodule SHAs against the
# image (its container fingerprint) and refuses to run on a mismatch, so a dependency
# change needs a new image. Code-only patches committed on top of the pin are fine.
if ! git -C "${NEMO_RL_DIR}" merge-base --is-ancestor "${NEMO_RL_COMMIT}" HEAD 2>/dev/null; then
    echo "WARNING: ${NEMO_RL_DIR} is not based on the pinned commit ${NEMO_RL_COMMIT:0:9}" >&2
fi
# ray.sub comes from the clone; it needs our patches (branch ${NEMO_RL_BRANCH}).
if ! grep -q RAY_SINGLE_SRUN "${NEMO_RL_DIR}/ray.sub"; then
    echo "ERROR: ${NEMO_RL_DIR}/ray.sub lacks the Alps patches (check out ${NEMO_RL_BRANCH})" >&2; exit 1
fi
if [[ "${CONTAINER_RUNTIME}" == pyxis ]] && ! grep -q PYXIS_EXTRA_ARGS "${NEMO_RL_DIR}/ray.sub"; then
    echo "ERROR: ${NEMO_RL_DIR}/ray.sub lacks PYXIS_EXTRA_ARGS (git am backends/nemo-rl/patches/0002-*.patch)" >&2; exit 1
fi

mkdir -p "${OUT}/slurm" "${OUT}/results" "${OUT}/logs"

SBATCH_FLAGS=(--nodes="${NODES}" --account="${ACCOUNT}" --partition="${PARTITION}"
              --time="${JOB_TIME}" --job-name="nrl-${RUN}" --output="${OUT}/slurm/%j-${RUN}.out")
[[ -n "${RESERVATION}" ]] && SBATCH_FLAGS+=(--reservation="${RESERVATION}")
[[ -n "${EXCLUDE_NODES}" ]] && SBATCH_FLAGS+=(--exclude="${EXCLUDE_NODES}")
[[ -n "${DEPENDENCY:-}" ]] && SBATCH_FLAGS+=(--dependency="${DEPENDENCY}")
[[ "${PARTITION}" == preemptable ]] && SBATCH_FLAGS+=(--requeue --open-mode=append)
# Site-mandated flags (sites/<SITE>.env), e.g. HANGANG's --gres, --exclusive and --comment.
read -r -a extra <<< "${SBATCH_EXTRA}"
SBATCH_FLAGS+=(${extra[@]+"${extra[@]}"})

# Runs on the Ray head inside the container, in /opt/nemo-rl (the clone mount; the EDF's
# workdir, while pyxis starts in the submit directory).
# The image driver venv directly, not `uv run`: the venvs are complete and read-only for us.
# driver_retry.sh restarts a driver that fails mid-run (DRIVER_RETRIES, default 2; 0 disables).
# RAY_DEDUP_LOGS=0: Ray otherwise folds similar lines from different workers into one
# "[repeated Nx]" line, which drops per-rank errors (NCCL's failing CUDA call) and the
# per-rank [gpumem] lines.
# Weights & Biases when WANDB_PROJECT is set. The key stays out of the command and the job's
# environment (ray.sub traces its commands): driver.sh reads it from WANDB_KEY_FILE.
WANDB_ARGS=""
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        export WANDB_KEY_FILE="${WANDB_KEY_FILE:-${HOME}/.wandb_api_key}"
        [[ -r "${WANDB_KEY_FILE}" ]] || { echo "ERROR: WANDB_PROJECT is set but neither WANDB_API_KEY nor ${WANDB_KEY_FILE} holds the W&B API key" >&2; exit 1; }
    fi
    WANDB_ARGS="logger.wandb_enabled=true logger.wandb.project=${WANDB_PROJECT} logger.wandb.name=${RUN}"
    [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_ARGS+=" ++logger.wandb.entity=${WANDB_ENTITY}"
fi
COMMAND="cd /opt/nemo-rl && RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0} DRIVER_RETRIES=${DRIVER_RETRIES:-2} bash ${HERE}/driver_retry.sh \
${DRIVER_PYTHON:-/opt/nemo_rl_venv/bin/python} ${SCRIPT} --config ${CONFIG} \
cluster.num_nodes=${NODES} cluster.gpus_per_node=${GPUS_PER_NODE} \
checkpointing.checkpoint_dir=${OUT}/results/${RUN} \
logger.log_dir=${OUT}/logs/${RUN} ${WANDB_ARGS} $*"

# Container: EDF names the installed ~/.edf file; pyxis gets CONTAINER, MOUNTS and
# PYXIS_EXTRA_ARGS, with the spec's environment exported here (sbatch passes it on).
case "${CONTAINER_RUNTIME}" in
    edf)   export EDF="${EDF_NAME}" ;;
    pyxis) unset EDF; container_pyxis_env ;;
    *)     echo "ERROR: unknown CONTAINER_RUNTIME '${CONTAINER_RUNTIME}' (edf or pyxis)" >&2; exit 1 ;;
esac

# RAY_SINGLE_SRUN=1: head + workers in one srun step (Slingshot network is per step).
# SRUN_MEM_BIND: keep host memory on the CPU NUMA nodes 0-3, off the GPUs' HBM.
export GPUS_PER_NODE RAY_SINGLE_SRUN="${RAY_SINGLE_SRUN:-1}" BASE_LOG_DIR="${OUT}/slurm" COMMAND \
    SRUN_MEM_BIND="${SRUN_MEM_BIND-verbose,mask_mem:0xf}" NUMA_WATCH="${NUMA_WATCH-${HERE}/numa_watch.sh}"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "== DRY_RUN: site ${SITE}, runtime ${CONTAINER_RUNTIME}; would run in ${OUT}/slurm:"
    printf '  sbatch'; printf ' %q' "${SBATCH_FLAGS[@]}" "${NEMO_RL_DIR}/ray.sub"; echo
    for v in EDF CONTAINER MOUNTS PYXIS_EXTRA_ARGS GPUS_PER_NODE RAY_SINGLE_SRUN SRUN_MEM_BIND NUMA_WATCH BASE_LOG_DIR COMMAND; do
        [[ -n "${!v:-}" ]] && echo "  ${v}=${!v}"
    done
    if [[ "${CONTAINER_RUNTIME}" == pyxis ]]; then
        echo "  container environment:"
        for v in $(sed -n 's/.*--container-env=\([^ ]*\).*/\1/p' <<< "${PYXIS_EXTRA_ARGS}" | tr , ' '); do echo "    ${v}=${!v}"; done
    fi
    exit 0
fi

# ray.sub writes logs and helper files into the submit dir, so submit from scratch.
cd "${OUT}/slurm"
submitted="$(sbatch "${SBATCH_FLAGS[@]}" "${NEMO_RL_DIR}/ray.sub")"
echo "${submitted}"
if [[ -n "${PREFLIGHT_OUT}" ]]; then mv "${PREFLIGHT_OUT}" "${OUT}/slurm/${submitted##* }-${RUN}.preflight.txt"; fi
