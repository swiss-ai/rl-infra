#!/bin/bash
# Train Apertus 2 Chonk with GRPO on Alps from a single task file:
#   bash run.sh <task.yaml> [overrides...]        e.g.  bash run.sh envs/gsm8k.yaml
# The task file describes the RL environment: dataset, prompt and reward. Everything else
# comes from a recipe in recipes/, by default vLLM + AdamW on 8 trainer and 4 inference nodes,
# run as a 5-hour preemptable job that saves the final weights. Every run first brings the
# setup up to date; the first one clones the sources and copies the 200 GB model, later ones
# take seconds. RECIPE, RUN, PARTITION, JOB_TIME and ACCOUNT override the defaults.
# TODO: choose the recipe from the task, e.g. from its context length.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ $# -ge 1 && -f "$1" ]] || { sed -n '2,9s/^# \{0,1\}//p' "$0"; exit 1; }
TASK="$(realpath "$1")"; shift
RECIPE="$(realpath "${RECIPE:-${ROOT}/recipes/apertus2-chonk-vllm-adamw.yaml}")"
NODES="$(yq r "${RECIPE}" cluster.num_nodes)"
RUN="${RUN:-$(basename "${TASK}" .yaml)-$(date +%Y%m%d-%H%M%S)}"
export ACCOUNT="${ACCOUNT:-infra01}" PARTITION="${PARTITION:-preemptable}" JOB_TIME="${JOB_TIME:-05:00:00}"

[[ -f "${ROOT}/site.env" ]] || cp "${ROOT}/site.env.example" "${ROOT}/site.env"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${APERTUS2_DIR}/env"
source "${ROOT}/site.env"
export SCRATCH
mkdir -p "${OUT}/configs"

LOG="${OUT}/setup.log"
echo "== setup (log: ${LOG})"
{ SKIP_VLLM_IMAGE=1 bash "${APERTUS2_DIR}/setup.sh" && bash "${ROOT}/experiments/grpo-chonk/stage_model.sh"; } \
    > "${LOG}" 2>&1 || { tail -n 20 "${LOG}"; echo "ERROR: setup failed, see ${LOG}" >&2; exit 1; }

# The task is merged over the recipe, so it can also override any recipe setting.
CFG="${OUT}/configs/${RUN}.yaml"
printf 'defaults:\n  - %s\n  - %s\n' "${RECIPE}" "${TASK}" > "${CFG}"
submitted="$(bash "${APERTUS2_DIR}/launch.sh" "${CFG}" "${NODES}" "${RUN}" "$@" | tee /dev/stderr | grep -oE 'Submitted batch job [0-9]+')"
jobid="${submitted##* }"
echo
echo "Run ${RUN} is job ${jobid}."
echo "  training log:   ${OUT}/slurm/${jobid}-logs/ray-driver.log"
echo "  metrics:        ${OUT}/logs/${RUN}"
echo "  final weights:  ${OUT}/results/${RUN}"
