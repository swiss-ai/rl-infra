#!/bin/bash
# Generation throughput of a run's vLLM setup without the trainer (genbench.py), as one job of <nodes> nodes:
#   bash shared/debug/genbench.sh <name> <nodes> <run config.yaml> [overrides...]
# <run config.yaml> is the config copy run.sh made for a run ($OUT/configs/<run>-<time>/config.yaml); the overrides
# change its generation setup (e.g. policy.generation.vllm_cfg.kv_cache_dtype=fp8). The model is the run's
# (MODEL=<staged export> to change it), the context CONTEXT (default 32768). BENCH_CASES, MAX_TOKENS, JOB_TIME and
# EXCLUDE_NODES pass through; results in $OUT/genbench/results.jsonl, the log in $OUT/slurm/<job>-genbench-<name>.out.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NAME=$1; NODES=$2; CFG=$3; shift 3
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"
source "${PIN_FILE}"
CONTEXT="${CONTEXT:-32768}"
MODEL="${MODEL:-$(grep -m1 -o 'policy.model_name=[^ ]*' "$(dirname "${CFG}")/setup.log" 2>/dev/null | cut -d= -f2 || true)}"
MODEL="${MODEL:-${SCRATCH}/models/megachonk-sft-v1-16k-iter_0000880}"
# The overlays a run on an SFT export gets (run.sh): apertus-common for the Apertus 2 format.
export PYDEPS="${PYDEPS}-apertus2" PYDEPS_APERTUS2="${SCRATCH}/nemo-rl-apertus2/apertus-common-${APERTUS_COMMON_COMMIT:0:9}"
export RL_USER_HOME="${HOME}" RL_DATA_DIR="${RL_DATA_DIR:-}"
export SCRIPT="${ROOT}/shared/debug/genbench.py" PREFLIGHT=0 DRIVER_RETRIES=0 WANDB_PROJECT=
export JOB_TIME="${JOB_TIME:-01:15:00}" BENCH_NAME="${NAME}" BENCH_CASES="${BENCH_CASES:-64x1024,128x1024,256x1024}"
export MAX_TOKENS="${MAX_TOKENS:-6144}"
exec bash "${ROOT}/backends/nemo-rl/apertus2/launch.sh" "${CFG}" "${NODES}" "genbench-${NAME}" \
    cluster.num_nodes="${NODES}" policy.generation.colocated.resources.num_nodes="${NODES}" \
    policy.model_name="${MODEL}" policy.max_total_sequence_length="${CONTEXT}" \
    "++policy.generation.vllm_cfg.env_vars.VLLM_ALLOW_LONG_MAX_MODEL_LEN=\"'1'\"" "$@"
