#!/bin/bash
# Put NeMo Gym with the OpenSandbox provider and the Harbor agent into the NeMo-RL clone that
# setup.sh made (safe to re-run). Run setup.sh first.
#
# The blueprint's NeMo-RL fork pins Gym at 267305e2, which has neither the OpenSandbox provider
# nor the Harbor training config. Those are four commits of Kostas Pitas on the swiss-ai/nemo-gym
# branch swissai/harbor_agent_osb_v0.7.0 (they predate the pin and apply on it without conflicts):
#   810196ff6  a rollout with no output items is masked, not trained on
#   0fe241c66  harbor_agent: derive CPU caps in the sandbox environment
#   44c77151b  harbor_agent: OpenSandbox config for RL training
#   85f6def4a  bug fixes
# The result is a local branch apertus2-osb in the clone's Gym checkout (never pushed).
#
# Optional: GYM_BRANCH (default apertus2-osb) names the local branch, GYM_PATCHES (space-separated
# patch files, default none) are applied with `git am` on top of the four commits when that branch
# is created, e.g. for experiments/harbor-capture-sc:
#   GYM_BRANCH=apertus2-osb-capture \
#   GYM_PATCHES="$PWD/backends/nemo-rl/apertus2/patches/gym/0001-*.patch" \
#   bash backends/nemo-rl/apertus2/setup-gym.sh
# The checkout is shared by every run of this NeMo-RL clone; switch back with GYM_BRANCH=apertus2-osb.
set -euo pipefail
APERTUS2_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${APERTUS2_DIR}/../../.." && pwd)"
source "${APERTUS2_DIR}/env"
source "${ROOT}/site.env"

GYM_URL="${GYM_URL:-https://github.com/swiss-ai/nemo-gym.git}"
GYM_PIN="${GYM_PIN:-267305e2a}"
GYM_OSB_BRANCH="${GYM_OSB_BRANCH:-swissai/harbor_agent_osb_v0.7.0}"
GYM_OSB_COMMITS="${GYM_OSB_COMMITS:-810196ff6 0fe241c66 44c77151b 85f6def4a}"
GYM_BRANCH="${GYM_BRANCH:-apertus2-osb}"
GYM_PATCHES="${GYM_PATCHES:-}"
GYM_DIR="${NEMO_RL_DIR}/3rdparty/Gym-workspace/Gym"
GIT=(git -c user.name="$(git -C "${ROOT}" config user.name || echo rl-infra)" -c user.email="$(git -C "${ROOT}" config user.email || echo rl-infra@localhost)" -c commit.gpgsign=false)

[[ -d "${NEMO_RL_DIR}/.git" ]] || { echo "ERROR: no NeMo-RL clone in ${NEMO_RL_DIR}; run setup.sh first" >&2; exit 1; }

if [[ ! -d "${GYM_DIR}/.git" ]]; then
    echo "== cloning Gym into ${GYM_DIR}"
    rm -rf "${GYM_DIR}"
    mkdir -p "$(dirname "${GYM_DIR}")"
    git clone --filter=blob:none "${GYM_URL}" "${GYM_DIR}"
fi
if git -C "${GYM_DIR}" rev-parse --verify -q "${GYM_BRANCH}" >/dev/null; then
    echo "== Gym branch ${GYM_BRANCH} exists ($(git -C "${GYM_DIR}" rev-parse --short "${GYM_BRANCH}"))"
else
    git -C "${GYM_DIR}" fetch origin "${GYM_OSB_BRANCH}"
    git -C "${GYM_DIR}" switch -c "${GYM_BRANCH}" "${GYM_PIN}"
    # shellcheck disable=SC2086
    "${GIT[@]}" -C "${GYM_DIR}" cherry-pick ${GYM_OSB_COMMITS}
    for patch in ${GYM_PATCHES}; do
        [[ -f "${patch}" ]] || { echo "ERROR: Gym patch ${patch} not found" >&2; exit 1; }
        echo "== applying $(basename "${patch}")"
        "${GIT[@]}" -C "${GYM_DIR}" am --3way "${patch}"
    done
fi
git -C "${GYM_DIR}" switch -q "${GYM_BRANCH}"
git -C "${GYM_DIR}" remote set-url --push origin no_push
# Gym needs Python 3.13 since 2026-08-04 (#2194), but David's image is Python 3.12, and Gym builds every
# server venv with the driver's Python (Ray needs the same version on both sides). The code runs on 3.12
# (the driver imports it), so relax the declared minimum in the working tree; without this the servers'
# `uv pip install` fails with "the current Python version (3.12.3) does not satisfy Python>=3.13.14".
grep -rl --include=pyproject.toml 'requires-python = ">=3.13.14"' "${GYM_DIR}" \
    | xargs -r sed -i 's/requires-python = ">=3.13.14"/requires-python = ">=3.12"/'
echo "== Gym ready: $(git -C "${GYM_DIR}" log --oneline -1)"
for f in nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
         responses_api_agents/harbor_agent/configs/harbor_agent_opensandbox_training.yaml; do
    [[ -f "${GYM_DIR}/${f}" ]] || { echo "ERROR: ${f} missing from the Gym checkout" >&2; exit 1; }
done
