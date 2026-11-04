#!/bin/bash
# Put NeMo Gym into the NeMo-RL clone that setup.sh made, for tasks that run on it (safe to re-run;
# run.sh calls it for a task that names a `script`). No image change: the Gym checkout goes on
# PYTHONPATH through the clone's mount, pydeps.sh adds what the image lacks, and Gym builds its
# server venvs at job start on scratch (NEMO_GYM_VENV_DIR in the container spec).
set -euo pipefail

APERTUS2_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${APERTUS2_DIR}/../../.." && pwd)"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"
source "${PIN_FILE}"
GYM_DIR="${NEMO_RL_DIR}/3rdparty/Gym-workspace/Gym"

[[ -d "${NEMO_RL_DIR}/.git" ]] || { echo "ERROR: no NeMo-RL clone in ${NEMO_RL_DIR}; run setup.sh first" >&2; exit 1; }

# The submodule path of the clone is an empty directory until this runs.
if [[ ! -d "${GYM_DIR}/.git" ]]; then
    echo "== cloning Gym into ${GYM_DIR}"
    git clone --filter=blob:none --no-checkout "${GYM_URL}" "${GYM_DIR}"
fi
if [[ "$(git -C "${GYM_DIR}" rev-parse HEAD 2>/dev/null)" != "${GYM_COMMIT}" ]]; then
    echo "== Gym at ${GYM_COMMIT:0:9}"
    git -C "${GYM_DIR}" checkout --detach "${GYM_COMMIT}"
fi
git -C "${GYM_DIR}" remote set-url --push origin no_push

# Our Gym patches go on the working tree (HEAD stays at the pin, so the check above holds). When
# they changed, the tree goes back to the pin first: new files of an older version would block them.
stamp="$(cat "${APERTUS2_DIR}"/patches/gym/*.patch | git hash-object --stdin)"
if [[ "$(cat "${GYM_DIR}/.git/rl-infra-patches" 2>/dev/null)" != "${stamp}" ]]; then
    git -C "${GYM_DIR}" checkout --quiet --force "${GYM_COMMIT}"
    # every file a patch creates (any version of it): untracked at the pin, so the checkout leaves it
    added="$(cat "${APERTUS2_DIR}"/patches/gym/*.patch | git apply --summary 2>/dev/null | awk '$1 == "create" {print $4}' | sort -u)"
    [[ -z "${added}" ]] || git -C "${GYM_DIR}" clean -fdxq -- ${added}
    git -C "${GYM_DIR}" clean -fdxq resources_servers/rubric_judge resources_servers/code_gen/opensandbox_pods.py
    for p in "${APERTUS2_DIR}"/patches/gym/*.patch; do
        git -C "${GYM_DIR}" apply "${p}"
        echo "== Gym: applied $(basename "${p}")"
    done
    echo "${stamp}" > "${GYM_DIR}/.git/rl-infra-patches"
fi

# Gym declares Python 3.13, but builds every server venv with the driver's Python, which is the
# image's 3.12 (Ray needs the same version on both sides). The code runs on 3.12; without this the
# servers' `uv pip install` fails with "the current Python version (3.12.3) does not satisfy".
{ grep -rl --include=pyproject.toml 'requires-python = ">=3.13.14"' "${GYM_DIR}" || true; } \
    | xargs -r sed -i 's/requires-python = ">=3.13.14"/requires-python = ">=3.12"/'
mkdir -p "${SCRATCH}/nemo-rl-apertus2/gym-venvs"
echo "== Gym ready: $(git -C "${GYM_DIR}" log --oneline -1)"
