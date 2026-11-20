#!/bin/bash
# Setup of the SFT backend on the login node (safe to re-run; sft.sh runs it before every submission):
#   1. clone swiss-ai/posttraining at the pin into POSTTRAINING_DIR and apply patches/, matched by subject, so a re-run
#      applies only the patches added since
#   2. stage the container image (the Apertus 2 NeMo-RL image) from its persistent master when scratch lacks it
# The Python overlay and the converted mixture need the image and are made by the first job (train.sh, check.sh).
set -euo pipefail

PT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${PT_DIR}/../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"   # the container image is the Apertus 2 NeMo-RL one
source "${ROOT}/sites/load.sh" && rl_infra_load "${PT_DIR}/env" "${APERTUS2_DIR}/env"
source "${PT_PIN_FILE}"
GIT=(git -c user.name="$(git -C "${ROOT}" config user.name || echo rl-infra)" -c user.email="$(git -C "${ROOT}" config user.email || echo rl-infra@localhost)" -c commit.gpgsign=false)

# 1. posttraining source
if [[ ! -e "${POSTTRAINING_DIR}/.git" ]]; then
    echo "== cloning posttraining into ${POSTTRAINING_DIR}"
    git clone --filter=blob:none "${POSTTRAINING_URL}" "${POSTTRAINING_DIR}"
    git -C "${POSTTRAINING_DIR}" switch -c rl-infra "${POSTTRAINING_COMMIT}"
    git -C "${POSTTRAINING_DIR}" remote set-url --push origin no_push
fi
git -C "${POSTTRAINING_DIR}" merge-base --is-ancestor "${POSTTRAINING_COMMIT}" HEAD 2>/dev/null || {
    echo "ERROR: ${POSTTRAINING_DIR} lacks the pinned commit ${POSTTRAINING_COMMIT:0:9}; move it aside for a fresh clone" >&2
    exit 1
}
applied="$(git -C "${POSTTRAINING_DIR}" log --format=%s)"
for p in "${PT_DIR}"/patches/*.patch; do
    subject="$(git mailinfo /dev/null /dev/null < "${p}" | sed -n 's/^Subject: //p')"
    grep -qxF -- "${subject}" <<< "${applied}" && continue
    echo "== posttraining: applying $(basename "${p}")"
    "${GIT[@]}" -C "${POSTTRAINING_DIR}" am -3 "${p}" || {
        git -C "${POSTTRAINING_DIR}" am --abort
        echo "ERROR: $(basename "${p}") does not apply; move ${POSTTRAINING_DIR} aside for a fresh clone" >&2
        exit 1
    }
done
echo "== posttraining: ${POSTTRAINING_DIR} ($(git -C "${POSTTRAINING_DIR}" rev-parse --short HEAD))"

# 2. container image: the scratch copy jobs read, from its persistent master (HANGANG reads the master in place)
if [[ ! -s "${IMAGE_WORK}" ]]; then
    [[ -s "${IMAGE_MASTER:-}" ]] || { echo "ERROR: no image at ${IMAGE_WORK} and no master to copy it from (IMAGE_MASTER)" >&2; exit 1; }
    echo "== image: copying ${IMAGE_MASTER} to ${IMAGE_WORK}"
    mkdir -p "$(dirname "${IMAGE_WORK}")"
    cp "${IMAGE_MASTER}" "${IMAGE_WORK}.tmp.$$"
    mv "${IMAGE_WORK}.tmp.$$" "${IMAGE_WORK}"
fi
echo "== image: ${IMAGE_WORK}"
