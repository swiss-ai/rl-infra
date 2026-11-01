#!/bin/bash
# Build what a sandbox sees under /sandbox: NumPy and LiveCodeBench's test runner. Safe to re-run.
set -euo pipefail

SANDBOX_DIR="$(cd "$(dirname "$0")" && pwd)"
: "${GYM_DIR:?set GYM_DIR to a Gym checkout}"
ENV_DIR="${SCRATCH}/code-sandbox/env"

mkdir -p "${ENV_DIR}/harness"
[[ -d "${ENV_DIR}/site/numpy" ]] \
    || "${UV:-uv}" pip install --python /usr/bin/python3.12 --target "${ENV_DIR}/site" numpy==2.1.0
cp "${GYM_DIR}/resources_servers/code_gen/lcb_integration/testing_util.py" \
    "${SANDBOX_DIR}/sandbox_harness.py" "${ENV_DIR}/harness/"
echo "== sandbox environment ready: ${ENV_DIR}"
