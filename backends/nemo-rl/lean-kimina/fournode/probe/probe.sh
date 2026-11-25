#!/bin/bash
# Stage-2 probe (fournode/README): the 4-node colocated recipe on the 200-step run's Lean setup for STEPS GRPO steps,
# with a checkpoint after step FIRST_STEPS and a resume from it inside the same job (probe_driver.sh). Records GPU
# memory per GPU, host memory per node, phase times and ScratchAdamW's step time; report.py reads them.
#
#   bash probe.sh setup         private copies of the source clones in $PROBE/src with Megatron 0007 and NeMo-RL
#                               0034 applied, and the probe's own EDF in $PROBE/edf; submits nothing
#   bash probe.sh env           prints the settings below as the job would get them; runs nothing
#   DRY_RUN=1 bash probe.sh     setup, preflight and the sbatch command (via run-200.sh); submits nothing
#   bash probe.sh               submits one 4-node job
#   python3 report.py $PROBE/nemo-rl <job>      after the job
#
# Nothing shared changes: the 200-step chain's clones (~/projects/...), its EDF (~/.edf/nemo-rl-apertus2.toml), its
# output root ($SCRATCH/nemo-rl-grpo200) and its Triton cache are only read. Setup copies the clones (rsync, read
# only on the source side); apertus2/setup.sh then patches the copies. The job writes only under $PROBE and
# $SCRATCH/lean-kimina/runs/<RUN> (the in-job Kimina server's endpoint and logs).
#
# Settings (environment):
#   PROBE            default $SCRATCH/lean/fournode-probe; must be under $SCRATCH
#   SHARED_PROJECTS  where the clones are copied from, default $HOME/projects
#   RUN              default fournode-probe-<date>-<time>
#   STEPS            GRPO steps in all, default 5; FIRST_STEPS the first driver's, default 3 (its checkpoint is the
#                    resume point)
#   VLLM_UTIL        vLLM's GPU memory share, default the recipe's 0.75 (stage 1's sleep/wake numbers may change it)
#   PARTITION        default normal (never preempted; 4 nodes do not fit debug's 90 node-minutes)
#   JOB_TIME         default 04:00:00: two start-ups of ~20 min, five steps of ~20-30 min, five checkpoints
#   R3               1: router replay on (policy.router_replay.enabled; Megatron patch 0008 for Chonk's quantile-balancing
#                    routers), with NeMo RL's route validation and an R3 trace of step 1 in logs/<RUN>/r3_trace (check it
#                    with tools/check_r3_trace.py). A probe without and one with R3 compare the train/inference mismatch.
#   EXTRA_OVERRIDES  appended after the probe's own (they win)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../../.." && pwd)"
LEAN_KIMINA_DIR="${ROOT}/backends/nemo-rl/lean-kimina"
STAGE2="${HERE}/../stage2"

[[ -n "${SCRATCH:-}" ]] || { echo "ERROR: SCRATCH is not set" >&2; exit 1; }
PROBE="${PROBE:-${SCRATCH}/lean/fournode-probe}"
case "${PROBE%/}/" in
    "${SCRATCH%/}"/?*) ;;
    *) echo "ERROR: PROBE ${PROBE} is not under SCRATCH ${SCRATCH}" >&2; exit 1 ;;
esac
SHARED_PROJECTS="${SHARED_PROJECTS:-${HOME}/projects}"
SRC="${PROBE}/src"
CLONES=(nemo-rl-apertus2 megatron-lm-moe-apertus2 megatron-bridge-apertus2 nemo-rl-apertus2-vllm)
STEPS="${STEPS:-5}" FIRST_STEPS="${FIRST_STEPS:-3}"
(( FIRST_STEPS >= 1 && FIRST_STEPS < STEPS )) || { echo "ERROR: need 1 <= FIRST_STEPS < STEPS" >&2; exit 1; }

# The private stack: apertus2/env and sites/load.sh keep a variable that is already set.
export NEMO_RL_DIR="${SRC}/nemo-rl-apertus2" MEGATRON_DIR="${SRC}/megatron-lm-moe-apertus2" \
    BRIDGE_DIR="${SRC}/megatron-bridge-apertus2" \
    VLLM_RUNNER="${SRC}/nemo-rl-apertus2-vllm/gpu_model_runner.py" VLLM_APERTUS2="${SRC}/nemo-rl-apertus2-vllm/apertus2.py" \
    EDF_NAME=nemo-rl-apertus2-fournode-probe EDF_DIR="${PROBE}/edf" OUT="${PROBE}/nemo-rl"
# The 200-step run's Lean and Mathlib (read only; Kimina stages them into node memory), as grpo200/settings.sh.
LEAN_ROOT=/capstor/store/cscs/swissai/infra01/lean_apertus_v2/lean-v4.34.0
export LEAN_PROJECT="${LEAN_ROOT}" LEAN_TOOLCHAIN="${LEAN_ROOT}/elan/toolchains/leanprover--lean4---v4.34.0"
# The job: run-200.sh's overrides on the 4-node recipe, the two-phase driver, the per-node watcher, no driver retry
# (a failure should end the probe, not hide in a retry).
export RUN="${RUN:-fournode-probe-$(date +%Y%m%d-%H%M%S)}" \
    RECIPE="${ROOT}/recipes/apertus2-chonk-sftv02-18315-vllm-adamw-code-4n-colocated.yaml" \
    PARTITION="${PARTITION:-normal}" JOB_TIME="${JOB_TIME:-04:00:00}" TRITON_CACHE="${PROBE}/triton" \
    DRIVER_PYTHON="bash ${HERE}/probe_driver.sh" PROBE_STEPS="${STEPS}" PROBE_FIRST_STEPS="${FIRST_STEPS}" \
    NUMA_WATCH="${HERE}/probe_watch.sh" DRIVER_RETRIES=0
# The task's Python overlays, as run.sh sets them for a Gym task on an SFT export (the EDF names them).
APERTUS_COMMON_COMMIT="$(source "${ROOT}/backends/nemo-rl/apertus2/pin" && echo "${APERTUS_COMMON_COMMIT}")"
export PYDEPS_GYM=1 PYDEPS="${SCRATCH}/nemo-rl-apertus2/pydeps-gym-apertus2" \
    PYDEPS_APERTUS2="${SCRATCH}/nemo-rl-apertus2/apertus-common-${APERTUS_COMMON_COMMIT:0:9}"
# run.sh's setup stays out (setup below did the probe's part; its model staging would re-link the shared model folder
# the running chain reads). TMPDIR is set for setup only: sbatch hands the environment to the job, where Ray would
# put its session folder and sockets under it.
export SKIP_SETUP=1
probe_overrides="grpo.max_num_steps=${STEPS} checkpointing.save_period=${FIRST_STEPS}"
[[ -z "${VLLM_UTIL:-}" ]] || probe_overrides+=" policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_UTIL}"
if [[ "${R3:-0}" == 1 ]]; then
    # '"1"': Ray takes only strings as environment values (see run-200.sh's Triton settings)
    probe_overrides+=" ++policy.router_replay.enabled=true"
    for kv in NRL_ROUTER_REPLAY_VALIDATE=1 NRL_R3_TRACE=1 NRL_R3_TRACE_VERIFY_FORWARD=1 NRL_R3_TRACE_STEPS=1; do
        probe_overrides+=" ++policy.megatron_cfg.env_vars.${kv%%=*}='\"${kv#*=}\"'"
    done
    probe_overrides+=" ++policy.megatron_cfg.env_vars.NRL_R3_TRACE_DIR=${OUT}/logs/${RUN}/r3_trace"
fi
export EXTRA_OVERRIDES="${probe_overrides}${EXTRA_OVERRIDES:+ ${EXTRA_OVERRIDES}}"

print_env() {
    for v in PROBE SHARED_PROJECTS NEMO_RL_DIR MEGATRON_DIR BRIDGE_DIR VLLM_RUNNER VLLM_APERTUS2 EDF_NAME EDF_DIR OUT \
             LEAN_PROJECT LEAN_TOOLCHAIN RUN RECIPE PARTITION JOB_TIME TRITON_CACHE DRIVER_PYTHON PROBE_STEPS \
             PROBE_FIRST_STEPS NUMA_WATCH DRIVER_RETRIES SKIP_SETUP PYDEPS PYDEPS_GYM PYDEPS_APERTUS2 \
             EXTRA_OVERRIDES; do
        printf '%s=%s\n' "${v}" "${!v}"
    done
}

# State of the shared clones (HEAD and working tree), to show that setup leaves them alone. --no-optional-locks:
# git status would otherwise refresh (write) the shared clone's index.
shared_state() {
    local c
    for c in nemo-rl-apertus2 megatron-lm-moe-apertus2 megatron-bridge-apertus2; do
        printf '%s %s %s\n' "${c}" "$(git -C "${SHARED_PROJECTS}/${c}" rev-parse HEAD)" \
            "$(git --no-optional-locks -C "${SHARED_PROJECTS}/${c}" status --porcelain | git hash-object --stdin)"
    done
    sha256sum "${SHARED_PROJECTS}"/nemo-rl-apertus2-vllm/* "${HOME}/.edf/nemo-rl-apertus2.toml" 2>/dev/null
}

# What run.sh's skipped setup would have made, already there from the 200-step run (only read here): the staged model,
# the apertus-common checkout and the Python overlay.
shared_inputs() {
    local export_dir model n n_index common overlay
    export_dir="$(sed -n 's/^sft_export: *//p' "${RECIPE}")"
    model="${SCRATCH}/models/$(basename "${export_dir}")"
    [[ -f "${model}/config.json" && -f "${model}/MODELING_SOURCE" ]] || { echo "ERROR: model not staged in ${model}" >&2; exit 1; }
    n=$(find "${model}/" -maxdepth 1 -name '*.safetensors' | wc -l)
    n_index=$(jq -r '.weight_map[]' "${model}/model.safetensors.index.json" | sort -u | wc -l)
    [[ "${n}" == "${n_index}" ]] || { echo "ERROR: ${model} has ${n} of ${n_index} shards" >&2; exit 1; }
    # setup.sh would check apertus-common out at the pin if it were elsewhere: it must be there already
    [[ "$(git -C "${PYDEPS_APERTUS2}" rev-parse HEAD 2>/dev/null)" == "${APERTUS_COMMON_COMMIT}" ]] \
        || { echo "ERROR: ${PYDEPS_APERTUS2} is not at ${APERTUS_COMMON_COMMIT:0:9}" >&2; exit 1; }
    [[ -d "${PYDEPS}" ]] || { echo "ERROR: no Python overlay ${PYDEPS}" >&2; exit 1; }
    echo "== shared inputs present: ${model} (${n} shards), ${PYDEPS_APERTUS2}, ${PYDEPS}"
}

setup() {
    local c before
    export TMPDIR="${PROBE}/tmp"
    mkdir -p "${SRC}" "${EDF_DIR}" "${OUT}" "${TMPDIR}"
    before="$(shared_state)"
    for c in "${CLONES[@]}"; do
        [[ -d "${SHARED_PROJECTS}/${c}" ]] || { echo "ERROR: no ${SHARED_PROJECTS}/${c}" >&2; exit 1; }
        if [[ -d "${SRC}/${c}" ]]; then echo "== ${c}: copy exists"; continue; fi
        echo "== copying ${SHARED_PROJECTS}/${c} to ${SRC}/${c}"
        rm -rf "${SRC}/${c}.part"
        rsync -a "${SHARED_PROJECTS}/${c}/" "${SRC}/${c}.part/"
        mv "${SRC}/${c}.part" "${SRC}/${c}"
    done
    # The Triton cache starts as a copy of the shared one (compiled kernels and autotuning results), then is the
    # probe's own: the running chain keeps writing to the shared one.
    local shared_triton="${SCRATCH}/lean/jit-cache/triton"
    if [[ ! -d "${TRITON_CACHE}" && -d "${shared_triton}" ]]; then
        echo "== seeding ${TRITON_CACHE} from ${shared_triton}"
        rm -rf "${TRITON_CACHE}.part" && cp -a "${shared_triton}" "${TRITON_CACHE}.part" && mv "${TRITON_CACHE}.part" "${TRITON_CACHE}"
    fi
    shared_inputs
    echo "== apertus2/setup.sh and setup-gym.sh on the copies (log: ${PROBE}/setup.log)"
    { SKIP_VLLM_IMAGE=1 bash "${ROOT}/backends/nemo-rl/apertus2/setup.sh" \
        && bash "${ROOT}/backends/nemo-rl/apertus2/setup-gym.sh"; } > "${PROBE}/setup.log" 2>&1 \
        || { tail -n 20 "${PROBE}/setup.log"; echo "ERROR: setup failed" >&2; exit 1; }
    grep -E "^== (NeMo-RL|Megatron-LM|EDF)" "${PROBE}/setup.log" || true
    # The patches reached the copies, and the EDF mounts them.
    # (the log goes into a variable first: grep -q ends the pipe early, and git's SIGPIPE fails it under pipefail)
    local subject applied
    subject="$(git mailinfo /dev/null /dev/null < "$(ls "${ROOT}"/backends/nemo-rl/apertus2/patches/nemo-rl/0034-*.patch)" | sed -n 's/^Subject: //p')"
    applied="$(git -C "${NEMO_RL_DIR}" log --format=%s)"
    grep -qxF -- "${subject}" <<< "${applied}" || { echo "ERROR: NeMo-RL 0034 is not in ${NEMO_RL_DIR}" >&2; exit 1; }
    grep -q cpu_optimizer_state_dir "${MEGATRON_DIR}/megatron/core/optimizer/optimizer_config.py" \
        || { echo "ERROR: Megatron 0007 is not in ${MEGATRON_DIR}" >&2; exit 1; }
    cmp -s "${STAGE2}/scratch_adam.py" "${MEGATRON_DIR}/megatron/core/optimizer/cpu_offloading/scratch_adam.py" \
        || { echo "ERROR: ${MEGATRON_DIR}'s scratch_adam.py is not stage2/scratch_adam.py" >&2; exit 1; }
    grep -qF "\"${NEMO_RL_DIR}:/opt/nemo-rl\"" "${EDF_DIR}/${EDF_NAME}.toml" && grep -qF "${MEGATRON_DIR}" "${EDF_DIR}/${EDF_NAME}.toml" \
        || { echo "ERROR: ${EDF_DIR}/${EDF_NAME}.toml does not mount the copies" >&2; exit 1; }
    # ... and is the chain's EDF with only the clone paths changed
    if ! diff <(sed "s|${SHARED_PROJECTS}/|${SRC}/|g" "${HOME}/.edf/nemo-rl-apertus2.toml") "${EDF_DIR}/${EDF_NAME}.toml"; then
        echo "ERROR: ${EDF_DIR}/${EDF_NAME}.toml differs from ~/.edf/nemo-rl-apertus2.toml beyond the clone paths" >&2
        exit 1
    fi
    [[ "$(shared_state)" == "${before}" ]] || { echo "ERROR: a shared clone or the shared EDF changed during setup" >&2; exit 1; }
    echo "== probe setup ready: patches 0007 + 0034 in ${SRC}, EDF ${EDF_DIR}/${EDF_NAME}.toml; shared clones unchanged"
}

case "${1:-}" in
    env) print_env; exit 0 ;;
    setup) setup; exit 0 ;;
    "") ;;
    *) sed -n '6,11s/^# \{0,1\}//p' "$0"; exit 1 ;;
esac
[[ -f "${EDF_DIR}/${EDF_NAME}.toml" && -d "${MEGATRON_DIR}" ]] || { echo "ERROR: run 'bash probe.sh setup' first" >&2; exit 1; }
before="$(shared_state)"
mkdir -p "${OUT}"
bash "${LEAN_KIMINA_DIR}/run-200.sh"
[[ "$(shared_state)" == "${before}" ]] || echo "WARNING: a shared clone or the shared EDF changed during the launch" >&2
