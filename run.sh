#!/bin/bash
# Train Apertus 2 Chonk with GRPO from a single task file (on any cluster with a sites/ profile):
#   bash run.sh <task.yaml> [overrides...]        e.g.  bash run.sh envs/gsm8k.yaml
# The task file describes the RL environment: dataset, prompt and reward. Everything else
# comes from a recipe in recipes/: the one the task names in `defaults:`, otherwise vLLM + AdamW
# on 8 trainer and 4 inference nodes, run as a 5-hour job (preemptable on Alps) that saves the
# final weights. Every run first brings the setup up to date; the first one clones the sources
# and stages the 200 GB model, later ones take seconds. RECIPE, RUN, PARTITION, JOB_TIME and
# ACCOUNT override the defaults; DRY_RUN=1 shows the job without submitting it. A task that names
# a `script` runs it instead of the default entry point; Gym tasks do, and get Gym set up too.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ $# -ge 1 && -f "$1" ]] || { sed -n '2,9s/^# \{0,1\}//p' "$0"; exit 1; }
TASK="$(realpath "$1")"; shift
TASK_RECIPE="$(yq r "${TASK}" defaults)"
[[ -z "${TASK_RECIPE}" ]] || TASK_RECIPE="$(dirname "${TASK}")/${TASK_RECIPE}"
RECIPE="$(realpath "${RECIPE:-${TASK_RECIPE:-${ROOT}/recipes/apertus2-chonk-vllm-adamw.yaml}}")"
NODES="$(yq r "${RECIPE}" cluster.num_nodes)"
RUN="${RUN:-$(basename "${TASK}" .yaml)-$(date +%Y%m%d-%H%M%S)}"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"
export ACCOUNT PARTITION="${PARTITION:-${RUN_PARTITION:-${DEFAULT_PARTITION}}}" JOB_TIME="${JOB_TIME:-05:00:00}" SCRATCH
mkdir -p "${OUT}/configs"
# pydeps.sh replaces an overlay when its list changes, and running jobs import from it, so a task
# with a script gets an overlay of its own.
TASK_SCRIPT="$(yq r "${TASK}" script)"
if [[ -n "${TASK_SCRIPT}" ]]; then
    export SCRIPT="${TASK_SCRIPT}" PYDEPS_GYM=1 PYDEPS="${SCRATCH}/nemo-rl-apertus2/pydeps-gym"
fi
# A recipe on an SFT checkpoint names its export. apertus-common goes in an overlay of its own, for
# the same reason, and the export is staged next to the base model.
SFT_EXPORT="$(yq r "${RECIPE}" sft_export)"
if [[ -n "${SFT_EXPORT}" ]]; then
    source "${PIN_FILE}"
    export PYDEPS_APERTUS2="${SCRATCH}/nemo-rl-apertus2/apertus-common-${APERTUS_COMMON_COMMIT:0:9}"
    export PYDEPS="${PYDEPS}-apertus2"
    export SRC="${SCRATCH}/${SFT_EXPORT}" DST="${SCRATCH}/models/$(basename "${SFT_EXPORT}")"
    export CHONK_MODEL_STAGE=copy STAGE_SHARDS=link
fi

LOG="${OUT}/setup.log"
echo "== setup (log: ${LOG})"
{ SKIP_VLLM_IMAGE=1 bash "${APERTUS2_DIR}/setup.sh" && bash "${ROOT}/experiments/grpo-chonk/stage_model.sh" \
    && { [[ -z "${TASK_SCRIPT}" ]] || bash "${APERTUS2_DIR}/setup-gym.sh"; }; } \
    > "${LOG}" 2>&1 || { tail -n 20 "${LOG}"; echo "ERROR: setup failed, see ${LOG}" >&2; exit 1; }

# The task is merged over the recipe, so it can also override any recipe setting. Its copy drops
# its own defaults, which would otherwise merge the recipe that RECIPE replaced over this one, and
# its script, which is not a config key.
CFG="${OUT}/configs/${RUN}.yaml"
yq d "${TASK}" defaults | yq d - script > "${OUT}/configs/${RUN}-task.yaml"
printf 'defaults:\n  - %s\n  - %s\n' "${RECIPE}" "${OUT}/configs/${RUN}-task.yaml" > "${CFG}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    exec bash "${APERTUS2_DIR}/launch.sh" "${CFG}" "${NODES}" "${RUN}" "$@"
fi
submitted="$(bash "${APERTUS2_DIR}/launch.sh" "${CFG}" "${NODES}" "${RUN}" "$@" | tee /dev/stderr | grep -oE 'Submitted batch job [0-9]+')"
jobid="${submitted##* }"
echo
echo "Run ${RUN} is job ${jobid}."
echo "  training log:   ${OUT}/slurm/${jobid}-logs/ray-driver.log"
echo "  metrics:        ${OUT}/logs/${RUN}"
echo "  final weights:  ${OUT}/results/${RUN}"
