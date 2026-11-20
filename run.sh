#!/bin/bash
# Train Apertus 2 Chonk with GRPO from a task file (on any cluster with a sites/ profile):
#   bash run.sh <task.yaml> [overrides...]        e.g.  bash run.sh envs/stem/main.yaml
# The task file says what to train on (datasets, environments, their shares of a step, steps,
# validation) and, with the rl-infra keys below, which recipe, context and node tier the run uses.
# Every launch first brings the setup up to date; the first one clones the sources and stages the
# model, later ones take seconds. Hydra overrides after the task file win over everything.
#
# Three ways to run it:
#   bash run.sh <task>             full run: submits the job, logs to W&B (refuses to start without it)
#   DEBUG=1 bash run.sh <task>     debug run: the name ends in -DEBUG, and unless the command line says
#                                  otherwise: 3 steps, no validation, no checkpoints, 2-hour limit
#   DRY_RUN=1 bash run.sh <task>   either of the two, shown instead of run: nodes, context, the overrides
#                                  made here, the config copies, preflight and the sbatch command; no
#                                  setup (it may change the NeMo-RL clone), no job
#
# rl-infra keys (top level of a task file, inherited through `defaults` like any other key):
#   recipe:    model and algorithm, a file under recipes/ (path relative to this repository)
#   context:   longest prompt + answer in tokens; sets CP, packing and memory from the first row of
#              backends/nemo-rl/apertus2/contexts.tsv that covers it (up to 131072)
#   size:      node tier, a file in sizes/ (cheap, expensive, ultra)
#   job_time:  Slurm time limit; a checkpointing run saves and stops 30 minutes before it
#   pydeps:    reasoning_gym, for Reasoning Gym's scorers in the math environment
#   script:    entry point instead of examples/run_grpo.py (NeMo Gym tasks)
#   template:  true in the placeholder files of a new expert; run.sh refuses to submit them
# A task without `recipe:` names its recipe in `defaults` (the older layout).
#
# Environment: RUN names the run (default: the task's path under envs/ and the time, e.g.
# stem-main-20261120-081500). RECIPE, SIZE and JOB_TIME replace the task's recipe, size and job_time.
# W&B: WANDB_PROJECT, WANDB_ENTITY and WANDB_KEY_FILE in site.env. PARTITION, ACCOUNT as usual.
#
# The job reads a copy of every config file made at launch (${OUT}/configs/<run>-<time>/), so an edit
# made while it waits in the queue, runs or retries does not reach it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ $# -ge 1 && -f "$1" ]] || { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 1; }
TASK="$(realpath "$1")"; shift
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"
source "${ROOT}/backends/nemo-rl/config.sh"
TMP="$(mktemp -d)"; trap 'rm -rf "${TMP}"' EXIT
# The NeMo-RL clone may not exist yet (setup.sh below makes it); its files hold no rl-infra keys.
export CONFIG_SKIP_MISSING_NEMO_RL=1

# --- rl-infra keys of the task
config_resolve "${TASK}" "${TMP}/task.yaml"
task_key() { local v; v="$(yq r "${TMP}/task.yaml" "$1")"; [[ "${v}" == null ]] && v=""; echo "${v}"; }
TASK_RECIPE="$(task_key recipe)"
CONTEXT="$(task_key context)"
SIZE="${SIZE:-$(task_key size)}"
DEBUG="${DEBUG:-0}"; DRY_RUN="${DRY_RUN:-0}"
# A debug run gets 2 hours unless JOB_TIME is given; the task's job_time is for its full run.
if [[ "${DEBUG}" == 1 ]]; then JOB_TIME="${JOB_TIME:-02:00:00}"; else JOB_TIME="${JOB_TIME:-$(task_key job_time)}"; fi
TASK_SCRIPT="$(task_key script)"
TASK_PYDEPS="$(task_key pydeps)"
# The placeholder files of a new expert carry `template: true` until their datasets are filled in.
if [[ "$(task_key template)" == true && "${DRY_RUN}" != 1 ]]; then
    echo "ERROR: ${TASK} is a template (template: true): fill in its datasets, then remove that key (DRY_RUN=1 works)" >&2; exit 1
fi

# --- run name and W&B
if [[ -z "${RUN:-}" ]]; then
    # envs/stem/dev/mix-a.yaml -> stem-dev-mix-a-<time>
    if [[ "${TASK}" == "${ROOT}/envs/"* ]]; then name="${TASK#"${ROOT}"/envs/}"; else name="$(basename "${TASK}")"; fi
    name="${name%.yaml}"
    RUN="${name//\//-}-$(date +%Y%m%d-%H%M%S)"
fi
[[ "${DEBUG}" == 1 && "${RUN}" != *-DEBUG ]] && RUN="${RUN}-DEBUG"
if [[ -z "${WANDB_PROJECT:-}" ]]; then
    [[ "${DEBUG}" == 1 ]] || { echo "ERROR: a full run must log to W&B: set WANDB_PROJECT in site.env (or DEBUG=1 for a debug run)" >&2; exit 1; }
    echo "WARNING: WANDB_PROJECT is not set: this debug run logs to TensorBoard only" >&2
fi
export RUN

# --- config: frozen copies of the recipe and task chains, merged by NeMo-RL in the job
DIR="${OUT}/configs/${RUN}-$(date +%Y%m%d-%H%M%S)"
parents=()
# A task that names no recipe at all (no `recipe:`, no `defaults`) gets vLLM + AdamW on the pretraining checkpoint.
if [[ -z "${RECIPE:-}" && -z "${TASK_RECIPE}" && -z "$(yq r "${TASK}" defaults)" ]]; then
    RECIPE="${ROOT}/recipes/apertus2-chonk-vllm-adamw.yaml"
fi
if [[ -n "${RECIPE:-}" || -n "${TASK_RECIPE}" ]]; then
    if [[ -n "${RECIPE:-}" ]]; then R="$(realpath "${RECIPE}")"; else R="${ROOT}/${TASK_RECIPE}"; fi
    [[ -f "${R}" ]] || { echo "ERROR: recipe ${R} not found" >&2; exit 1; }
    config_freeze "${R}" "${DIR}"; parents+=("${FROZEN}")
    # Older layout: the task names its recipe in `defaults`, which RECIPE replaces.
    if [[ -z "${TASK_RECIPE}" ]]; then config_freeze "${TASK}" "${DIR}" nodefaults; else config_freeze "${TASK}" "${DIR}"; fi
else
    R="(the task's defaults)"
    config_freeze "${TASK}" "${DIR}"
fi
parents+=("${FROZEN}")
CFG="${DIR}/config.yaml"
{ echo "defaults:"; printf '  - %s\n' "${parents[@]}"; } > "${CFG}"
config_resolve "${CFG}" "${TMP}/merged.yaml"
cfg() { local v; v="$(yq r "${TMP}/merged.yaml" "$1")"; [[ "${v}" == null ]] && v=""; echo "${v}"; }
OVR=()   # overrides made here; the command line's come after them and win

# --- nodes: the tier, or the recipe's own counts
JUDGE=$(( $(cfg env.judge_pool.num_nodes || true) + 0 ))
for n in $(yq r "${TMP}/merged.yaml" 'env.extra_judge_pools.*.num_nodes' 2>/dev/null); do JUDGE=$(( JUDGE + n )); done
if [[ -n "${SIZE}" ]]; then
    SF="${ROOT}/sizes/${SIZE}.yaml"
    [[ -f "${SF}" ]] || { echo "ERROR: no node tier sizes/${SIZE}.yaml" >&2; exit 1; }
    TRAIN=$(yq r "${SF}" trainer_nodes); GEN=$(yq r "${SF}" generation_nodes); SCALE=$(yq r "${SF}" batch_scale)
    NODES=$(( TRAIN + GEN + JUDGE ))
    OVR+=(cluster.num_nodes="${NODES}" policy.generation.colocated.resources.num_nodes="${GEN}")
    if (( SCALE != 1 )); then
        OVR+=(grpo.num_prompts_per_step=$(( $(cfg grpo.num_prompts_per_step) * SCALE ))
              policy.train_global_batch_size=$(( $(cfg policy.train_global_batch_size) * SCALE )))
        for k in $(yq r "${TMP}/merged.yaml" --printMode p 'data.num_prompts_per_dataloader.*' 2>/dev/null); do
            OVR+=("${k}=$(( $(cfg "${k}") * SCALE ))")
        done
    fi
else
    NODES="$(cfg cluster.num_nodes)"
    GEN=0; [[ "$(cfg policy.generation.colocated.enabled)" == false ]] && GEN="$(cfg policy.generation.colocated.resources.num_nodes)"
    TRAIN=$(( NODES - GEN - JUDGE ))
fi

# --- context: CP, packing and memory from the table
if [[ -n "${CONTEXT}" ]]; then
    read -r _ CP SCHEME PACK MEMFRAC < <(awk -v c="${CONTEXT}" '!/^#/ && NF == 5 && $1 >= c { print; exit }' "${APERTUS2_DIR}/contexts.tsv") || true
    [[ -n "${CP:-}" ]] || { echo "ERROR: context ${CONTEXT} is above the longest one in contexts.tsv" >&2; exit 1; }
    PP="$(cfg policy.megatron_cfg.pipeline_model_parallel_size)"; RANKS=$(( TRAIN * GPUS_PER_NODE / ${PP:-1} ))
    (( RANKS % CP == 0 )) || { echo "ERROR: CP ${CP} (context ${CONTEXT}) does not divide the ${RANKS} trainer ranks of a pipeline stage" >&2; exit 1; }
    OVR+=(policy.max_total_sequence_length="${CONTEXT}" policy.generation.max_new_tokens="${CONTEXT}"
          policy.sequence_packing.train_mb_tokens="${PACK}" policy.sequence_packing.logprob_mb_tokens="${PACK}"
          policy.megatron_cfg.context_parallel_size="${CP}" policy.make_sequence_length_divisible_by=$(( CP > 1 ? 2 * CP : 2 ))
          "++policy.megatron_cfg.env_vars.NRL_CUDA_MEMORY_FRACTION=\"'${MEMFRAC}'\""
          # vLLM refuses a limit above the export's max_position_embeddings (16,384 for the SFT checkpoints)
          # unless this is set; the model has no position table, so the value is only that check.
          "++policy.generation.vllm_cfg.env_vars.VLLM_ALLOW_LONG_MAX_MODEL_LEN=\"'1'\"")
    if [[ "${SCHEME}" == kcp ]]; then
        source "${APERTUS2_DIR}/kcp/use.sh"
        OVR+=(++policy.megatron_cfg.model_overrides.linear_attention_cp_impl=kcp)
    fi
fi

# --- debug run: short and without validation or checkpoints, unless the command line says otherwise
if [[ "${DEBUG}" == 1 ]]; then
    OVR+=(grpo.max_num_steps=3 grpo.val_at_start=false grpo.val_at_end=false grpo.val_period=1000
          checkpointing.enabled=false)
fi

# --- time limit, and the checkpoint deadline before it
JOB_TIME="${JOB_TIME:-05:00:00}"
if [[ "$(cfg checkpointing.enabled)" == true ]]; then
    t="${JOB_TIME}"; d=0; [[ "${t}" == *-* ]] && { d=${t%%-*}; t=${t#*-}; }
    IFS=: read -r h m s <<< "${t}"
    left=$(( d * 86400 + 10#${h} * 3600 + 10#${m} * 60 + 10#${s:-0} - 1800 ))
    (( left > 0 )) || { echo "ERROR: JOB_TIME ${JOB_TIME} leaves no time to save a checkpoint" >&2; exit 1; }
    # Quoted for Hydra, which reads an unquoted colon as a separator.
    OVR+=("checkpointing.checkpoint_must_save_by=\"'$(printf '%02d:%02d:%02d:%02d' $(( left / 86400 )) $(( left % 86400 / 3600 )) $(( left % 3600 / 60 )) $(( left % 60 )))'\"")
fi
export ACCOUNT PARTITION="${PARTITION:-${RUN_PARTITION:-${DEFAULT_PARTITION}}}" JOB_TIME SCRATCH
# HOME inside the container is not yours; task files find your files under ${oc.env:RL_USER_HOME} and the
# prepared datasets under ${oc.env:RL_DATA_DIR} (sites/<SITE>.env).
export RL_USER_HOME="${HOME}" RL_DATA_DIR="${RL_DATA_DIR:-}"
if [[ -z "${RL_DATA_DIR}" ]] && grep -q 'RL_DATA_DIR' "${DIR}"/*.yaml; then
    echo "ERROR: the task reads \${oc.env:RL_DATA_DIR}; set RL_DATA_DIR (sites/<SITE>.env or site.env)" >&2; exit 1
fi
mkdir -p "${OUT}/configs"

# --- runtime overlays
# pydeps.sh replaces an overlay when its list changes, and running jobs import from it, so a task
# with a script gets an overlay of its own. The task alone sets the overlay flags: one left exported
# in the shell would put its packages into the overlay of a task that does not ask for them.
unset PYDEPS_GYM PYDEPS_RG
if [[ -n "${TASK_SCRIPT}" ]]; then
    export SCRIPT="${TASK_SCRIPT}" PYDEPS_GYM=1 PYDEPS="${SCRATCH}/nemo-rl-apertus2/pydeps-gym"
    # Gym builds its server venvs with uv and keeps uv's cache (about 70k files per build) in the
    # directory NeMo-RL gives it: the container's `uv cache dir`, unless the config names one. NeMo-RL's
    # ray.sub unsets UV_CACHE_DIR before srun, so under pyxis (HANGANG) the container spec's value never
    # arrives and the cache fell back to $XDG_CACHE_HOME/uv on scratch, whatever JOB_UV_CACHE_DIR said.
    # Naming it in the config works on every site; it goes first, so a command-line override still wins.
    OVR=("++env.nemo_gym.uv_cache_dir=${JOB_UV_CACHE_DIR:-${SCRATCH}/.cache/uv}" "${OVR[@]}")
fi
# Reasoning Gym scored in NeMo-RL's math environment gets reasoning-gym in an overlay of its own.
if [[ "${TASK_PYDEPS}" == reasoning_gym ]]; then
    export PYDEPS_RG=1 PYDEPS="${PYDEPS}-rg"
elif [[ -n "${TASK_PYDEPS}" ]]; then
    echo "ERROR: unknown pydeps '${TASK_PYDEPS}' in ${TASK} (known: reasoning_gym)" >&2; exit 1
fi
# A recipe on an SFT checkpoint names its export. apertus-common goes in an overlay of its own, for
# the same reason, and the export is staged next to the base model.
SFT_EXPORT="$(cfg sft_export)"
if [[ -n "${SFT_EXPORT}" ]]; then
    source "${PIN_FILE}"
    export PYDEPS_APERTUS2="${SCRATCH}/nemo-rl-apertus2/apertus-common-${APERTUS_COMMON_COMMIT:0:9}"
    export PYDEPS="${PYDEPS}-apertus2"
    export SRC="${SCRATCH}/${SFT_EXPORT}" DST="${SCRATCH}/models/$(basename "${SFT_EXPORT}")"
    # An export that already carries the modeling code (MODELING_SOURCE) is linked whole and only read;
    # stage_model.sh checks its hfconverter commit and auto_map.
    if [[ -f "${SRC}/MODELING_SOURCE" ]]; then
        export CHONK_MODEL_STAGE=link
    else
        export CHONK_MODEL_STAGE=copy STAGE_SHARDS=link
    fi
fi

echo "Task:    ${TASK}"
echo "Recipe:  ${R}"
echo "Nodes:   ${NODES} = ${TRAIN} trainer + ${GEN} vLLM + ${JUDGE} judge${SIZE:+ (tier ${SIZE})}"
[[ -n "${CONTEXT}" ]] && echo "Context: ${CONTEXT} tokens: CP ${CP} (${SCHEME}), packing ${PACK}, memory fraction ${MEMFRAC}"
echo "Run:     ${RUN} ($([[ "${DEBUG}" == 1 ]] && echo debug || echo full) run, time limit ${JOB_TIME})"
echo "Config:  ${CFG}"
echo "Overrides made here (the command line's come after them and win):"
printf '  %s\n' "${OVR[@]}"

# --- dry run: preflight and the sbatch command, without setup (which may change the NeMo-RL clone)
if [[ "${DRY_RUN}" == 1 ]]; then
    [[ -e "${NEMO_RL_DIR}/.git" ]] || {
        echo "ERROR: no NeMo-RL clone at ${NEMO_RL_DIR} yet: launch once without DRY_RUN, or run backends/nemo-rl/apertus2/setup.sh" >&2; exit 1; }
    exec bash "${APERTUS2_DIR}/launch.sh" "${CFG}" "${NODES}" "${RUN}" "${OVR[@]}" "$@"
fi

LOG="${OUT}/setup.log"
echo "== setup (log: ${LOG})"
{ SKIP_VLLM_IMAGE=1 bash "${APERTUS2_DIR}/setup.sh" && bash "${ROOT}/experiments/grpo-chonk/stage_model.sh" \
    && { [[ -z "${TASK_SCRIPT}" ]] || bash "${APERTUS2_DIR}/setup-gym.sh"; }; } \
    > "${LOG}" 2>&1 || { tail -n 20 "${LOG}"; echo "ERROR: setup failed, see ${LOG}" >&2; exit 1; }
cp "${LOG}" "${DIR}/setup.log"

submitted="$(bash "${APERTUS2_DIR}/launch.sh" "${CFG}" "${NODES}" "${RUN}" "${OVR[@]}" "$@" | tee /dev/stderr | grep -oE 'Submitted batch job [0-9]+')"
jobid="${submitted##* }"
echo
echo "Run ${RUN} is job ${jobid}."
echo "  training log:   ${OUT}/slurm/${jobid}-logs/ray-driver.log"
echo "  config copies:  ${DIR}"
echo "  metrics:        ${OUT}/logs/${RUN}"
echo "  checkpoints:    ${OUT}/results/${RUN}"
