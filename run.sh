#!/bin/bash
# RL training on rl-infra: one RL expert, or any task file (on any cluster with a sites/ profile):
#   bash run.sh <expert> [overrides...]           e.g.  bash run.sh stem     (experts/stem/base.yaml)
#   bash run.sh <task.yaml> [overrides...]        e.g.  bash run.sh experts/stem/dev/<variant>.yaml
# An expert's base.yaml says what to train on (datasets, environments, their shares of a step,
# validation) and, with the rl-infra keys below, which recipe, context and node tier its run uses.
# Under it run.sh puts experts/main.yaml, the run settings every expert shares (checkpoints, rollout
# dump, validation period, default tier and time); a task in the older layout (recipe in `defaults`,
# the files in archive/) does not take it and runs as it ran.
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
#   context:   longest prompt + answer in tokens (up to 131072); with the recipe's context_parallel block it sets
#              CP (context / tokens_per_gpu, rounded up to a power of 2), packing (= context), KCP above the recipe's head_cp_max
#              and the memory fraction
#   size:      nodes: a tier of the recipe's base layout (`nodes:` in the recipe; Chonk 8 trainer x 8 vLLM):
#              lite (base trainer, half the vLLM), cheap (the base), expensive (twice the base), ultra
#              (expensive with twice the vLLM); or explicit counts, <trainer>+<vLLM> (e.g. 16+32)
#   job_time:  Slurm time limit; a checkpointing run saves and stops 30 minutes before it
#   pydeps:    reasoning_gym, for Reasoning Gym's scorers in the math environment
#   script:    entry point instead of examples/run_grpo.py (NeMo Gym tasks)
#   template:  true in the placeholder files of a new expert; run.sh refuses to submit them
# A task without `recipe:` names its recipe in `defaults` (the older layout).
#
# Environment: RUN names the run (default: the task's path under experts/ and the time, e.g.
# stem-20261120-081500, stem-dev-<variant>-20261120-081500). RECIPE, SIZE and JOB_TIME replace the
# task's recipe, size and job_time.
# W&B: WANDB_PROJECT, WANDB_ENTITY and WANDB_KEY_FILE in site.env. PARTITION, ACCOUNT as usual.
#
# The job reads a copy of every config file made at launch (${OUT}/configs/<run>-<time>/), so an edit
# made while it waits in the queue, runs or retries does not reach it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 1; }
[[ $# -ge 1 ]] || usage
# An expert name or folder stands for that expert's base.yaml.
if [[ -f "$1" ]]; then TASK="$1"
elif [[ -d "$1" && -f "$1/base.yaml" ]]; then TASK="$1/base.yaml"
elif [[ -f "${ROOT}/experts/$1/base.yaml" ]]; then TASK="${ROOT}/experts/$1/base.yaml"
else echo "ERROR: $1 is neither a task file nor an expert ($(cd "${ROOT}/experts" && ls -d */ | tr -d / | tr '\n' ' '))" >&2; exit 1
fi
TASK="$(realpath "${TASK}")"; shift
# The run settings every expert shares; taken by tasks in the expert layout (with a `recipe:` key).
MAIN="${ROOT}/experts/main.yaml"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"
source "${ROOT}/backends/nemo-rl/config.sh"
TMP="$(mktemp -d)"; trap 'rm -rf "${TMP}"' EXIT
# The NeMo-RL clone may not exist yet (setup.sh below makes it); its files hold no rl-infra keys.
export CONFIG_SKIP_MISSING_NEMO_RL=1

# --- rl-infra keys of the task (with experts/main.yaml under it in the expert layout)
config_resolve "${TASK}" "${TMP}/task.yaml"
task_key() { local v; v="$(yq r "${TMP}/task.yaml" "$1")"; [[ "${v}" == null ]] && v=""; echo "${v}"; }
TASK_RECIPE="$(task_key recipe)"
if [[ -n "${TASK_RECIPE}" ]]; then
    printf 'defaults:\n  - %s\n  - %s\n' "${MAIN}" "${TASK}" > "${TMP}/with-main.yaml"
    config_resolve "${TMP}/with-main.yaml" "${TMP}/task.yaml"
fi
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
    # experts/stem/base.yaml -> stem-<time>, experts/stem/dev/mix-a.yaml -> stem-dev-mix-a-<time>
    if [[ "${TASK}" == "${ROOT}/experts/"* ]]; then name="${TASK#"${ROOT}"/experts/}"; else name="$(basename "${TASK}")"; fi
    name="${name%.yaml}"; name="${name%/base}"
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
    RECIPE="${ROOT}/recipes/archive/apertus2-chonk-vllm-adamw.yaml"
fi
if [[ -n "${RECIPE:-}" || -n "${TASK_RECIPE}" ]]; then
    if [[ -n "${RECIPE:-}" ]]; then R="$(realpath "${RECIPE}")"; else R="${ROOT}/${TASK_RECIPE}"; fi
    [[ -f "${R}" ]] || { echo "ERROR: recipe ${R} not found" >&2; exit 1; }
    config_freeze "${R}" "${DIR}"; parents+=("${FROZEN}")
    if [[ -n "${TASK_RECIPE}" ]]; then
        # Expert layout: recipe, then the shared run settings, then the expert's files. NeMo-RL applies a file's
        # `_override_: true` only against that file's own `defaults` chain (and OmegaConf refuses to merge a list
        # into a dict), so the recipe and main.yaml become parents of the task chain instead of siblings in
        # config.yaml: they go into the `defaults` of its first file without parents (e.g. shared/judges/
        # llm-judge.yaml under stem/base.yaml), and base.yaml's `data: {_override_: true}` replaces the recipe's.
        config_freeze "${MAIN}" "${DIR}"; MAIN_FROZEN="${FROZEN}"
        config_freeze "${TASK}" "${DIR}"
        leaf="${FROZEN}"
        while first="$(_config_defaults "${leaf}" | head -1)"; [[ -n "${first}" ]]; do leaf="$(_config_parent "${leaf}" "${first}")"; done
        { printf 'defaults:\n  - %s\n  - %s\n' "${parents[0]}" "${MAIN_FROZEN}"; cat "${leaf}"; } > "${leaf}.tmp"
        mv "${leaf}.tmp" "${leaf}"
        parents=()
    else
        # Older layout: the task names its recipe in `defaults`, which RECIPE replaces.
        config_freeze "${TASK}" "${DIR}" nodefaults
    fi
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

# --- nodes: a tier of the recipe's base layout (nodes: in the recipe), explicit counts, or the task's own
JUDGE=$(( $(cfg env.judge_pool.num_nodes || true) + 0 ))
for n in $(yq r "${TMP}/merged.yaml" 'env.extra_judge_pools.*.num_nodes' 2>/dev/null); do JUDGE=$(( JUDGE + n )); done
if [[ -n "${SIZE}" ]]; then
    if [[ "${SIZE}" =~ ^([0-9]+)\+([0-9]+)$ ]]; then
        TRAIN="${BASH_REMATCH[1]}"; GEN="${BASH_REMATCH[2]}"
    else
        BT="$(cfg nodes.trainer)"; BI="$(cfg nodes.inference)"
        [[ -n "${BT}" && -n "${BI}" ]] || { echo "ERROR: size ${SIZE} needs the recipe's base layout (nodes: trainer, inference); give the counts as SIZE=<trainer>+<vLLM> instead" >&2; exit 1; }
        # The tiers, from the recipe's base layout (trainer x inference nodes; Chonk 8 x 8):
        #   lite = base trainer, half the inference; cheap = base; expensive = twice the base;
        #   ultra = expensive with twice the inference.
        case "${SIZE}" in
            lite)      TRAIN="${BT}";         GEN=$(( BI / 2 )) ;;
            cheap)     TRAIN="${BT}";         GEN="${BI}" ;;
            expensive) TRAIN=$(( 2 * BT ));   GEN=$(( 2 * BI )) ;;
            ultra)     TRAIN=$(( 2 * BT ));   GEN=$(( 4 * BI )) ;;
            *) echo "ERROR: size ${SIZE}: a tier (lite, cheap, expensive, ultra) or <trainer>+<vLLM> nodes, e.g. 16+32" >&2; exit 1 ;;
        esac
    fi
    NODES=$(( TRAIN + GEN + JUDGE ))
    OVR+=(cluster.num_nodes="${NODES}" policy.generation.colocated.resources.num_nodes="${GEN}")
else
    NODES="$(cfg cluster.num_nodes)"
    GEN=0; [[ "$(cfg policy.generation.colocated.enabled)" == false ]] && GEN="$(cfg policy.generation.colocated.resources.num_nodes)"
    TRAIN=$(( NODES - GEN - JUDGE ))
fi

# --- context: CP, packing and memory from the recipe's context_parallel block
#   tokens_per_gpu   most tokens of a packed micro-batch one trainer GPU holds: CP is context / tokens_per_gpu,
#                    rounded up to a power of 2 (Chonk 8192: 16K -> CP 2, 32K -> CP 4, 128K -> CP 16)
#   memory_fraction  NRL_CUDA_MEMORY_FRACTION, PyTorch's share of a trainer GPU (the rest: NCCL, Triton, CUDA);
#                    not set when the recipe leaves it out
#   head_cp_max      the largest CP at which the pinned Megatron splits the KDA layers' heads over the CP ranks: the
#                    KDA heads per TP rank (Chonk 28 heads at TP1: 4; MegaChonk 64 at TP4: 16); default 4. Above it
#                    the KDA layers need KCP (backends/nemo-rl/apertus2/kcp).
# Packing budget = context.
if [[ -n "${CONTEXT}" ]]; then
    (( CONTEXT <= 131072 )) || { echo "ERROR: context ${CONTEXT} is above 131072" >&2; exit 1; }
    TPG="$(cfg context_parallel.tokens_per_gpu)"
    [[ -n "${TPG}" ]] || { echo "ERROR: context: needs context_parallel.tokens_per_gpu in the recipe" >&2; exit 1; }
    CP=1; while (( CP * TPG < CONTEXT )); do CP=$(( CP * 2 )); done
    PACK="${CONTEXT}"; MEMFRAC="$(cfg context_parallel.memory_fraction)"
    PP="$(cfg policy.megatron_cfg.pipeline_model_parallel_size)"; RANKS=$(( TRAIN * GPUS_PER_NODE / ${PP:-1} ))
    TP="$(cfg policy.megatron_cfg.tensor_model_parallel_size)"
    (( RANKS % (CP * ${TP:-1}) == 0 )) || { echo "ERROR: TP ${TP:-1} x CP ${CP} (context ${CONTEXT}) does not divide the ${RANKS} trainer ranks of a pipeline stage: fewer pipeline stages or more trainer nodes" >&2; exit 1; }
    HEADCP="$(cfg context_parallel.head_cp_max)"; HEADCP="${HEADCP:-4}"
    # Every packed sequence is padded to a multiple of 2 x CP, times TP with sequence parallelism (NeMo-RL's check).
    PAD=$(( CP > 1 ? 2 * CP : 2 ))
    [[ "$(cfg policy.megatron_cfg.sequence_parallel)" == true && "${TP:-1}" -gt 1 ]] && PAD=$(( PAD * TP ))
    OVR+=(policy.max_total_sequence_length="${CONTEXT}" policy.generation.max_new_tokens="${CONTEXT}"
          policy.sequence_packing.train_mb_tokens="${PACK}" policy.sequence_packing.logprob_mb_tokens="${PACK}"
          policy.megatron_cfg.context_parallel_size="${CP}" policy.make_sequence_length_divisible_by="${PAD}"
          # vLLM refuses a limit above the export's max_position_embeddings (16,384 for the SFT checkpoints)
          # unless this is set; the model has no position table, so the value is only that check.
          "++policy.generation.vllm_cfg.env_vars.VLLM_ALLOW_LONG_MAX_MODEL_LEN=\"'1'\"")
    [[ -z "${MEMFRAC}" ]] || OVR+=("++policy.megatron_cfg.env_vars.NRL_CUDA_MEMORY_FRACTION=\"'${MEMFRAC}'\"")
    if (( CP > HEADCP )); then
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
# A recipe on an SFT checkpoint names its export (absolute, or relative to $SCRATCH; a symlink to an
# export elsewhere works). apertus-common goes in an overlay of its own, for the same reason. The export
# is staged as $SCRATCH/models/<name>: its shards are linked, not copied; only the small files are copied
# when the modeling code has to be added to them. policy.model_name is set to that path here, so a recipe
# names the model once.
SFT_EXPORT="$(cfg sft_export)"
if [[ -n "${SFT_EXPORT}" ]]; then
    source "${PIN_FILE}"
    export PYDEPS_APERTUS2="${SCRATCH}/nemo-rl-apertus2/apertus-common-${APERTUS_COMMON_COMMIT:0:9}"
    export PYDEPS="${PYDEPS}-apertus2"
    if [[ "${SFT_EXPORT}" == /* ]]; then SRC="${SFT_EXPORT}"; else SRC="${SCRATCH}/${SFT_EXPORT}"; fi
    export SRC DST="${SCRATCH}/models/$(basename "${SFT_EXPORT}")"
    OVR+=(policy.model_name="${DST}")
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
[[ -n "${CONTEXT}" ]] && echo "Context: ${CONTEXT} tokens: CP ${CP} ($( (( CP > HEADCP )) && echo kcp || echo head)), packing ${PACK}, memory fraction ${MEMFRAC:-not set}"
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
{ SKIP_VLLM_IMAGE=1 bash "${APERTUS2_DIR}/setup.sh" && bash "${ROOT}/backends/nemo-rl/apertus2/stage_model.sh" \
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
