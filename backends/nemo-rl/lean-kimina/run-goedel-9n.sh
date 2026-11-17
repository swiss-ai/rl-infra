#!/bin/bash
# The Lean GRPO run of 2026-11-17 (Clariden job 3965303), as one command: Chonk SFT v0.2 on 2,357 Goedel-Prover
# problems, 9 nodes, 12 h, the Kimina server inside the job. RUNBOOK.md ("The Goedel run") says why each setting.
#   DRY_RUN=1 bash backends/nemo-rl/lean-kimina/run-goedel-9n.sh      setup, preflight and the sbatch command
#   bash backends/nemo-rl/lean-kimina/run-goedel-9n.sh [overrides...]  submits; overrides go after this run's
# Needs what every train.sh run needs (RUNBOOK.md, "Once per account"), including validation.v434.jsonl from
# setup-data.sh. The training rows are data/goedel-train-s1.v434.jsonl.gz (made by goedel_rows.py, see RUNBOOK.md);
# they are unpacked to $LEAN_DATA/data once and checked against the run's sha256.
# RUN (default lean-kimina-v02-goedel-9n-<date>-<time>), PARTITION and JOB_TIME work as in train.sh.
set -euo pipefail
LEAN_KIMINA_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${LEAN_KIMINA_DIR}/../../.." && pwd)"
source "${ROOT}/sites/load.sh" && rl_infra_load "${LEAN_KIMINA_DIR}/env"
ROWS="${LEAN_DATA}/data/goedel-train-s1.v434.jsonl"
SHA256=47e7b60208716969386980210fa0530a30b17588d7f44690bb8e75f69e7e4f40
if [[ ! -s "${ROWS}" ]]; then
    mkdir -p "$(dirname "${ROWS}")"
    gunzip -c "${LEAN_KIMINA_DIR}/data/goedel-train-s1.v434.jsonl.gz" > "${ROWS}.part" && mv "${ROWS}.part" "${ROWS}"
fi
echo "${SHA256}  ${ROWS}" | sha256sum -c --quiet \
    || { echo "ERROR: ${ROWS} is not this run's training rows (sha256 ${SHA256})" >&2; exit 1; }
[[ -s "${LEAN_DATA}/data/validation.v434.jsonl" ]] \
    || { echo "ERROR: no ${LEAN_DATA}/data/validation.v434.jsonl; run setup-data.sh first" >&2; exit 1; }

cd "${ROOT}"
export RUN="${RUN:-lean-kimina-v02-goedel-9n-$(date +%Y%m%d-%H%M%S)}" HEADER_ROWS="${ROWS}" \
    RECIPE=recipes/apertus2-chonk-sftv02-18315-vllm-adamw-code-9n.yaml
exec bash "${LEAN_KIMINA_DIR}/train.sh" \
    data.train.data_path="${ROWS}" \
    grpo.num_prompts_per_step=32 grpo.val_period=20 \
    checkpointing.save_period=10 checkpointing.keep_top_k=null \
    policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.max_new_tokens=16000 \
    env.nemo_gym.math_formal_lean.resources_servers.math_formal_lean.compilation_timeout=120 \
    "$@"
