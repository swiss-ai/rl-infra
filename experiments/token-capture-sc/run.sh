#!/bin/bash
# Token-capture test, Single-Controller variant: mini_swe_agent_2 on OpenSandbox with Qwen3-4B on
# one node, through NeMo-RL's examples/run_grpo_single_controller.py with token capture at the
# model endpoint. Same data, venv and litellm handling as ../token-capture-test/run.sh, which it
# calls. See README.md.
#   bash experiments/token-capture-sc/run.sh [overrides...]
# Env: as ../token-capture-test/run.sh.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
export TASK="${HERE}/mini-swe-osb-sc.yaml" RECIPE="${HERE}/qwen3-4b-1n-sc.yaml"
export SCRIPT=examples/run_grpo_single_controller.py
export RUN="${RUN:-token-test-sc-$(date +%Y%m%d-%H%M%S)}"
export NEMO_GYM_WORKLOAD="${NEMO_GYM_WORKLOAD:-token-capture-sc}"
exec bash "${HERE}/../token-capture-test/run.sh" "$@"
