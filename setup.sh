#!/bin/bash
# One-time setup on the login node (safe to re-run):
#   1. clone NeMo-RL at the pinned commit, with submodules, if it is not there yet
#   2. on EDF sites (Alps), render the EDF into ~/.edf/<EDF_NAME>.toml (the old one is backed up
#      if it differs)
#   3. stage the container image working copy on scratch
# Settings come from sites/load.sh (cluster profile, optional site.env).
# Nothing is pip-installed: the NGC image ships every venv, prebuilt for the pinned commit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
source "${ROOT}/sites/load.sh" && rl_infra_load
source "${PIN_FILE}"
source "${ROOT}/backends/nemo-rl/container.sh"

# 1. NeMo-RL source
if [[ ! -d "${NEMO_RL_DIR}/.git" ]]; then
    echo "== cloning NeMo-RL into ${NEMO_RL_DIR}"
    git clone "${NEMO_RL_URL}" "${NEMO_RL_DIR}"
    git -C "${NEMO_RL_DIR}" switch -c "${NEMO_RL_BRANCH}" "${NEMO_RL_COMMIT}"
    # Recursive: Megatron-LM is nested inside Megatron-Bridge; without it the Megatron
    # backend fails with "No module named 'megatron.core'".
    git -C "${NEMO_RL_DIR}" submodule update --init --recursive --depth 1
    # Our patches (Alps launcher) as commits on the local branch.
    git -C "${NEMO_RL_DIR}" am "${ROOT}"/backends/nemo-rl/patches/*.patch
    # Read-only clone: guard against accidental pushes to the upstream mirror.
    git -C "${NEMO_RL_DIR}" remote set-url --push origin no_push
else
    echo "== NeMo-RL clone exists: ${NEMO_RL_DIR} ($(git -C "${NEMO_RL_DIR}" branch --show-current) @ $(git -C "${NEMO_RL_DIR}" rev-parse --short HEAD))"
    if ! git -C "${NEMO_RL_DIR}" merge-base --is-ancestor "${NEMO_RL_COMMIT}" HEAD; then
        echo "WARNING: clone is not based on the pinned commit ${NEMO_RL_COMMIT:0:9}" >&2
    fi
    if ! grep -q RAY_SINGLE_SRUN "${NEMO_RL_DIR}/ray.sub"; then
        echo "WARNING: clone's ray.sub lacks the Alps patches; check out ${NEMO_RL_BRANCH} or git am backends/nemo-rl/patches/*.patch" >&2
    fi
fi

# 2. Container: the EDF on Alps; pyxis flags are built from the same spec at launch.
if [[ "${CONTAINER_RUNTIME}" == edf ]]; then
    container_install_edf
else
    echo "== container runtime ${CONTAINER_RUNTIME}: nothing to install (launch.sh builds the flags)"
fi

# 3. Image working copy (striped for many-node opens)
if [[ ! -s "${IMAGE_WORK}" ]]; then
    [[ "${IMAGE_WORK}" != "${IMAGE_MASTER}" && -s "${IMAGE_MASTER}" ]] \
        || { echo "ERROR: image ${IMAGE_WORK} missing and no master to stage it from" >&2; exit 1; }
    echo "== staging image ${IMAGE_WORK}"
    mkdir -p "$(dirname "${IMAGE_WORK}")"
    lfs setstripe -c 16 -S 4M "$(dirname "${IMAGE_WORK}")" 2>/dev/null || true
    # Per-process temp name + rename, same as backends/nemo-rl/launch.sh.
    tmp="${IMAGE_WORK}.tmp.$$"
    cp "${IMAGE_MASTER}" "${tmp}" || { rm -f "${tmp}"; echo "ERROR: image copy failed" >&2; exit 1; }
    mv "${tmp}" "${IMAGE_WORK}"
else
    echo "== image present: ${IMAGE_WORK}"
fi
