#!/bin/bash
# Check B of OpenCode on the SFT model (research/research_notes/2026-11-10-apertus2-tool-calls-for-opencode.md
# in the alpernates workspace), CPU only, in the run S container: (1) NeMo-RL's Apertus 2 format tests
# (patch 0025 and later) with the pinned apertus-common and the SFT tokenizer; (2) the request files in
# check_apertus2_tools/ encoded as the chat endpoint encodes them: prompt length, tool cost, room left
# in the context, rendered prompt, and one tool round trip. opencode-1.17.11-first-request.json is the
# body OpenCode 1.17.11 sends first under our Gym adapter (cwd /app, task micro-fib-091, model id = the
# run S model path with USER for the user), rebuilt from OpenCode's source; the same rebuild for the
# Qwen3-4B id gives exactly the 5406 prompt tokens job 5314432 measured (with its cwd /).
# Runs as a one-node job, after run S's setup (setup.sh with the apertus-common overlay into the SFT
# EDF, stage_model.sh):
#   EDF_NAME=nemo-rl-apertus2-sft CHECK_SCRIPT=check_apertus2_tools.sh CHECK_CPUS=8 JOB_TIME=00:20:00 \
#       bash backends/nemo-rl/apertus2/check_env_job.sh
# The SFT EDF sets PYDEPS and PYDEPS_APERTUS2. MODEL (default the run S model), SYSTEM_PROMPT_FILE
# (default envs/apertus2-system.json, as envs/harbor-osb-sft.yaml) and CONTEXT (default 16384, run S
# policy.max_total_sequence_length) override.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
MODEL="${MODEL:-${SCRATCH}/models/chonk-lc-16k-v0.1-ep1-iter_0019374}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${ROOT}/envs/apertus2-system.json}"
CONTEXT="${CONTEXT:-16384}"
PY=/usr/bin/python3.12

: "${PYDEPS_APERTUS2:?not set: use the SFT EDF (EDF_NAME=nemo-rl-apertus2-sft)}"
bash "${HERE}/pydeps.sh"
[[ -r "${MODEL}/tokenizer.json" ]] || { echo "ERROR: no ${MODEL}/tokenizer.json (stage_model.sh)" >&2; exit 1; }

cd /opt/nemo-rl
echo "== NeMo-RL $(git log -1 --format='%h %s' 2>/dev/null || echo '(no git)'); apertus-common ${PYDEPS_APERTUS2}"
# The tests importorskip apertus_common: without it they would all be skipped and pass.
"${PY}" -c 'import apertus_common, tokenizers; print("apertus_common", apertus_common.__file__, "tokenizers", tokenizers.__version__)'

# --noconftest: tests/unit/conftest.py starts a Ray cluster for every session; these tests need none.
echo "== tests/unit/data/test_apertus2_format.py"
tests=0
NRL_APERTUS2_TOKENIZER="${MODEL}" "${PY}" -m pytest tests/unit/data/test_apertus2_format.py \
    --noconftest -p no:cacheprovider -o addopts="" -q -rs || tests=$?

echo "== OpenCode requests"
requests=0
"${PY}" "${HERE}/check_apertus2_tools.py" "${MODEL}" "${SYSTEM_PROMPT_FILE}" "${CONTEXT}" \
    "${HERE}"/check_apertus2_tools/*.json || requests=$?

echo "== tests exit ${tests}, requests exit ${requests}"
(( tests == 0 && requests == 0 ))
