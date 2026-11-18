#!/bin/bash
# One-time setup for KCP context parallelism (KCP=1 in launch-code-ctx.sh and launch-stem-ctx.sh): a Megatron checkout
# at the pinned commit with the KCP path of Megatron-LM-MoE PR 112 on branch rl-kcp, and flash-linear-attention 0.5.2
# (fla.ops.cp), which the launchers put ahead of the copy in the image. Plain copies only: run it on the login node or
# the data mover. The launchers pin Megatron to the rl-kcp commit, so setup.sh leaves that checkout as it is.
#   KCP_MEGATRON_DIR (default ${PROJECTS_DIR:-$HOME/projects}/megatron-lm-moe-apertus2-kcp)
#   KCP_FLA_DIR      (default ${PROJECTS_DIR:-$HOME/projects}/fla-0.5.2)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/backends/nemo-rl/apertus2/pin"
DIR=${KCP_MEGATRON_DIR:-${PROJECTS_DIR:-${HOME}/projects}/megatron-lm-moe-apertus2-kcp}
FLA=${KCP_FLA_DIR:-${PROJECTS_DIR:-${HOME}/projects}/fla-0.5.2}
if [[ ! -d "${DIR}/.git" ]]; then
    git clone "${MEGATRON_URL}" "${DIR}"
fi
git -C "${DIR}" fetch origin "${MEGATRON_REF}"
git -C "${DIR}" cat-file -e "${MEGATRON_COMMIT}^{commit}"
git -C "${DIR}" checkout -B rl-kcp "${MEGATRON_COMMIT}"
git -C "${DIR}" -c user.name="${GIT_AUTHOR_NAME:-kcp-setup}" -c user.email="${GIT_AUTHOR_EMAIL:-kcp-setup@localhost}" \
    am --committer-date-is-author-date "${HERE}"/kcp/*.patch
echo "== Megatron with KCP: ${DIR} at $(git -C "${DIR}" rev-parse --short rl-kcp)"
if [[ ! -d "${FLA}/fla" ]]; then
    "${UV:-uv}" pip install --python /usr/bin/python3.12 --no-deps --target "${FLA}" \
        flash-linear-attention==0.5.2 fla-core==0.5.2
fi
echo "== flash-linear-attention 0.5.2: ${FLA}"
