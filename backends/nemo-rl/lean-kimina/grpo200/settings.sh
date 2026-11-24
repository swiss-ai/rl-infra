# Settings of the 200-step GRPO run and its job chain (sourced by start.sh, guard.sh, stop.sh, status.sh).
# Only tests override SCRATCH_ROOT, CHAIN and RUN200.
SCRATCH_ROOT=${SCRATCH_ROOT:-/iopsstor/scratch/cscs/${USER}}
if [[ -z "${SCRATCH:-}" || "$(readlink -f "${SCRATCH}")" != "$(readlink -f "${SCRATCH_ROOT}")" ]]; then
    echo "ERROR: SCRATCH (${SCRATCH:-unset}) is not ${SCRATCH_ROOT}" >&2; exit 1
fi
RUN=lean-kimina-v02-grpo200-128x8
MAX_STEPS=200
# The rl-infra clone the chain's jobs run from, this run's own: update it (git pull/checkout) only while none of
# the chain's jobs is in the queue, or a queued job starts with the new code.
RL_INFRA=${RL_INFRA:-${SCRATCH}/rl-infra-grpo200}
RUN200=${RUN200:-${RL_INFRA}/backends/nemo-rl/lean-kimina/run-200.sh}
# The run's own output root, instead of $SCRATCH/nemo-rl (which other runs and the store copy use): configs/,
# slurm/, results/<RUN>/, logs/<RUN>/ and setup.log go here. Move them into $SCRATCH/nemo-rl after the run.
export OUT=${SCRATCH}/nemo-rl-grpo200
# Lean and Mathlib from the permanent install (identical to $SCRATCH/chonk-proofnet's: same Lean build 293d5d0,
# same Mathlib 5ed29652 and compiled files). Kimina stages them into node memory at every job start.
LEAN_ROOT=/capstor/store/cscs/swissai/infra01/lean_apertus_v2/lean-v4.34.0
export LEAN_PROJECT=${LEAN_ROOT} LEAN_TOOLCHAIN=${LEAN_ROOT}/elan/toolchains/leanprover--lean4---v4.34.0
# This folder: chain.tsv (the submitted jobs), guard.tsv (each guard's decision), logs/, STOP, DONE.
CHAIN=${CHAIN:-${SCRATCH}/lean/grpo200}
GUARD_NAME=grpo200-guard
TRAIN_NAME=nrl-${RUN}

# The newest complete checkpoint's step (NeMo-RL writes tmp_step_<N> and renames it to step_<N> when complete);
# 0 if none.
latest_step() {
    local d n best=0
    for d in "${OUT}/results/${RUN}"/step_*; do
        [[ -d "$d" ]] || continue
        n=${d##*/step_}
        if [[ "$n" =~ ^[0-9]+$ ]] && (( 10#$n > best )); then best=$((10#$n)); fi
    done
    echo "${best}"
}
# The rows of the newest chain in chain.tsv: "k training_job guard_job partition submitted".
chain_rows() { [[ -f "${CHAIN}/chain.tsv" ]] && awk '/^# chain started/ {delete r; n=0; next} {r[++n]=$0} END {for (i=1;i<=n;i++) print r[i]}' "${CHAIN}/chain.tsv"; }
