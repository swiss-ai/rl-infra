#!/bin/bash
# Submit a NeMo-RL run on Alps: ray.sub + the EDF installed by setup.sh.
#
# Usage: bash backends/nemo-rl/launch.sh <config> <num_nodes> <run_name> [overrides...]
#   <config>     a YAML in this repo (e.g. experiments/grpo-math/qwen3-32b-4n4g-fsdp2tp2.yaml),
#                any other path on a mounted filesystem, or an upstream recipe relative to
#                the NeMo-RL clone (e.g. examples/configs/grpo_math_1B.yaml)
#   overrides    passed to the training script, e.g. grpo.max_num_steps=50
#   SCRIPT=examples/run_vlm_grpo.py for vision-language models (default: examples/run_grpo.py)
#   PARTITION, JOB_TIME, ACCOUNT, USE_RESERVATION, EXCLUDE_NODES override site.env.
#
# Everything a job writes (Slurm/Ray logs, checkpoints, metrics) goes under $OUT on
# scratch; this repo and the NeMo-RL clone are only read.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
[[ -f "${ROOT}/site.env" ]] || { echo "ERROR: ${ROOT}/site.env missing (copy site.env.example)" >&2; exit 1; }
source "${ROOT}/site.env"
source "${HERE}/pin"

CONFIG=$1; NODES=$2; RUN=$3; shift 3
SCRIPT="${SCRIPT:-examples/run_grpo.py}"
PARTITION="${PARTITION:-${DEFAULT_PARTITION}}"
JOB_TIME="${JOB_TIME:-02:00:00}"
# No reservation by default; USE_RESERVATION=<name> to use one.
RESERVATION="${USE_RESERVATION:-}"

# The driver runs inside the container (cwd /opt/nemo-rl), so the config must be an
# absolute path it can see: /users and /iopsstor are mounted at the same paths, the
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

# Re-stage the image if the scratch purge removed it. Copy to a per-process temp name,
# then rename: the real path only ever holds a complete image, and two launches that
# both find it missing each finish their own copy (the last rename wins) instead of
# interleaving writes into one file.
if [[ ! -s "${IMAGE_WORK}" ]]; then
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
# ray.sub comes from the clone; it needs our Alps patches (branch ${NEMO_RL_BRANCH}).
if ! grep -q RAY_SINGLE_SRUN "${NEMO_RL_DIR}/ray.sub"; then
    echo "ERROR: ${NEMO_RL_DIR}/ray.sub lacks the Alps patches (check out ${NEMO_RL_BRANCH})" >&2; exit 1
fi

mkdir -p "${OUT}/slurm" "${OUT}/results" "${OUT}/logs"

SBATCH_FLAGS=(--nodes="${NODES}" --account="${ACCOUNT}" --partition="${PARTITION}"
              --time="${JOB_TIME}" --job-name="nrl-${RUN}" --output="${OUT}/slurm/%j-${RUN}.out")
[[ -n "${RESERVATION}" ]] && SBATCH_FLAGS+=(--reservation="${RESERVATION}")
[[ -n "${EXCLUDE_NODES}" ]] && SBATCH_FLAGS+=(--exclude="${EXCLUDE_NODES}")

# Runs on the Ray head inside the container, cwd = /opt/nemo-rl (the clone mount).
# The image driver venv directly, not `uv run`: the venvs are complete and read-only for us.
COMMAND="/opt/nemo_rl_venv/bin/python ${SCRIPT} --config ${CONFIG} \
cluster.num_nodes=${NODES} cluster.gpus_per_node=4 \
checkpointing.checkpoint_dir=${OUT}/results/${RUN} \
logger.log_dir=${OUT}/logs/${RUN} $*"

# ray.sub writes logs and helper files into the submit dir, so submit from scratch.
cd "${OUT}/slurm"
# RAY_SINGLE_SRUN=1: head + workers in one srun step (Slingshot network is per step).
# SRUN_MEM_BIND: keep host memory on the CPU NUMA nodes 0-3, off the GPUs' HBM.
EDF="${EDF_NAME}" GPUS_PER_NODE=4 RAY_SINGLE_SRUN="${RAY_SINGLE_SRUN:-1}" BASE_LOG_DIR="${OUT}/slurm" COMMAND="${COMMAND}" \
SRUN_MEM_BIND="${SRUN_MEM_BIND-verbose,mask_mem:0xf}" NUMA_WATCH="${NUMA_WATCH-${HERE}/numa_watch.sh}" \
    sbatch "${SBATCH_FLAGS[@]}" "${NEMO_RL_DIR}/ray.sub"
