#!/bin/bash
# Install the MT resource-accounting patch on top of the pinned NeMo-RL clone.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
source "${ROOT}/sites/load.sh" && rl_infra_load
PATCH="${HERE}/patches/0001-grpo-reserve-mt-reward-resources.patch"
if git -C "${NEMO_RL_DIR}" apply --reverse --check "${PATCH}" 2>/dev/null; then
    echo "== MT resource accounting already installed"
else
    git -C "${NEMO_RL_DIR}" apply --check "${PATCH}"
    git -C "${NEMO_RL_DIR}" am "${PATCH}"
fi
