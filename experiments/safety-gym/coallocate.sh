#!/usr/bin/env bash
# Submit SafetyGym training and its native SML judge in one Slurm heterogeneous job.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
mode="${1:---help}"
case "$mode" in
  --submit|--test-only|--allocation) shift ;;
  --help|-h)
    echo 'Usage: [env from resume.sh] bash coallocate.sh --submit|--test-only RUN [Hydra overrides...]'
    exit 0 ;;
  *) echo 'Expected --submit, --test-only or --allocation.' >&2; exit 2 ;;
esac
RUN="${1:-}"; (($# == 0)) || shift
[[ "$RUN" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Pass a valid RUN name.' >&2; exit 2; }
export RUN SAFETY_PROFILE="${SAFETY_PROFILE:-adaptive}"
export SAFETY_RESUME="${SAFETY_RESUME:-0}"
[[ "$SAFETY_PROFILE" != legacy ]] || { echo 'Coallocation is only supported for adaptive profiles.' >&2; exit 2; }
export JOB_TIME=74:00:00
export CHECKPOINT_DEADLINE=03:00:00:00
export SAFETY_ADAPTIVE_JUDGE_WAIT_SECONDS=3600
export SAFETY_JUDGE_TIME=74:00:00 SAFETY_JUDGE_MAX_JOB_TIME=74:00:00
source "${HERE}/environment.sh"
source "${HERE}/../../sites/load.sh"
rl_infra_load "${APERTUS2_DIR}/env"

render_dir="${SAFETY_ADAPTIVE_ROOT}/runtime/sml-judge-coalloc-${RUN}"
judge_master="${render_dir}/master.sh"

if [[ "$mode" == --submit || "$mode" == --test-only ]]; then
  bash "${HERE}/run.sh" --check "$RUN" "$@"
  if [[ "${SAFETY_RESUME}" != 1 ]]; then
    [[ ! -e "${SAFETY_ADAPTIVE_JUDGE_ENDPOINT_FILE}" ]] || {
      echo "Refusing existing judge endpoint metadata: ${SAFETY_ADAPTIVE_JUDGE_ENDPOINT_FILE}" >&2; exit 2; }
    [[ ! -e "${SAFETY_ADAPTIVE_TRAINING_JOB_FILE}" ]] || {
      echo "Refusing existing training job metadata: ${SAFETY_ADAPTIVE_TRAINING_JOB_FILE}" >&2; exit 2; }
  fi
  mkdir -p "${OUT}/slurm"
  if [[ ! -s "$judge_master" ]]; then
    SAFETY_SML_RENDER_DIR="$render_dir" bash "${HERE}/judge.sh" --sml-coalloc-render
  fi
  [[ -s "$judge_master" ]] || { echo "Missing rendered SML master: $judge_master" >&2; exit 1; }
  grep -Fq "$RUN" "$judge_master" && grep -Fq "${SAFETY_ADAPTIVE_JUDGE_ENDPOINT_FILE}" "$judge_master" || {
    echo "Rendered judge does not match RUN or endpoint path: $judge_master" >&2; exit 1; }
  bash -n "$judge_master"
  test_only=()
  [[ "$mode" != --test-only ]] || test_only+=(--test-only)
  read -r -a resource_flags <<< "${SBATCH_EXTRA}"
  submit_args=(--parsable --export=ALL --job-name="nrl-${RUN}" --output="${OUT}/slurm/%j-${RUN}-coalloc.out"
    --chdir="${OUT}/slurm"
    --nodes=12 --account="${ACCOUNT}" --partition="${RUN_PARTITION}" --time="${JOB_TIME}")
  submit_args+=("${resource_flags[@]}")
  submit_args+=(: --nodes=1 --account="${ACCOUNT}" --partition="${RUN_PARTITION}" --time="${JOB_TIME}")
  submit_args+=("${resource_flags[@]}")
  sbatch_bin="${SBATCH_BIN:-$(command -v sbatch || true)}"
  [[ -n "$sbatch_bin" && -x "$sbatch_bin" ]] || { echo 'sbatch not found.' >&2; exit 2; }
  "$sbatch_bin" "${test_only[@]}" "${submit_args[@]}" "$HERE/coallocate.sh" --allocation "$RUN" "$@"
  exit $?
fi

# Validate fresh metadata before starting the SML publisher, which can create the endpoint immediately.
bash "${HERE}/run.sh" --check "$RUN" "$@"
if [[ "${SAFETY_RESUME}" != 1 ]]; then
  [[ ! -e "${SAFETY_ADAPTIVE_JUDGE_ENDPOINT_FILE}" && ! -e "${SAFETY_ADAPTIVE_TRAINING_JOB_FILE}" ]] || {
    echo 'Fresh coallocation found existing run metadata; choose a new RUN or resume with SAFETY_RESUME=1.' >&2; exit 2; }
fi

# Allocation component 0 is the 12-node native NeMo-RL batch context. Component 1
# is the one-node SML service. Its rendered native srun commands explicitly select 1.
[[ -s "$judge_master" ]] || { echo "Rendered SML master is missing: $judge_master" >&2; exit 1; }
mapfile -t judge_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_1:?Missing judge component node list}")
(( ${#judge_nodes[@]} == 1 )) || { echo "Expected one judge node, got ${#judge_nodes[@]}." >&2; exit 1; }
judge_node_list="${SLURM_JOB_NODELIST_HET_GROUP_1}"
judge_log="${OUT}/slurm/${SLURM_JOB_ID}-${RUN}-judge.out"
(
  unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX RAY_ADDRESS
  export SLURM_NODELIST="$judge_node_list"
  export SLURM_JOB_NODELIST="$judge_node_list"
  export SLURM_NNODES="${#judge_nodes[@]}"
  export SLURM_JOB_NUM_NODES="${#judge_nodes[@]}"
  export SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION_HET_GROUP_1:-${RUN_PARTITION}}"
  export SLURM_JOB_ACCOUNT="${SLURM_JOB_ACCOUNT_HET_GROUP_1:-${ACCOUNT}}"
  judge_workdir="${SML_WORKING_DIR:-${SCRATCH}/sml}"
  mkdir -p "$judge_workdir"
  cd "$judge_workdir"
  exec bash "$judge_master"
) >"$judge_log" 2>&1 &
judge_pid=$!
export IN_ALLOCATION=1
bash "${HERE}/run.sh" --submit "$RUN" "$@" &
train_pid=$!
if wait -n "$judge_pid" "$train_pid"; then
  first_status=0
else
  first_status=$?
fi
if kill -0 "$judge_pid" 2>/dev/null; then
  # Training ended first; the judge is no longer needed.
  train_status="$first_status"
  kill "$judge_pid" 2>/dev/null || true
  wait "$judge_pid" 2>/dev/null || true
  exit "$train_status"
fi
# The judge controller exited first; stop the other component to avoid wasting allocation time.
scancel "${SLURM_JOB_ID}"
wait "$train_pid" 2>/dev/null || true
exit "${first_status:-1}"
