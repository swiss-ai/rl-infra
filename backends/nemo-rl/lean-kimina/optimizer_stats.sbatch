#!/bin/bash
#SBATCH --job-name=optimizer-stats
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --array=0-1
# Adam moments (step 40) and fp32 movement (step 30 -> 40) of the eps 1e-14 run and of H1 (eps 1e-8), per tensor and
# per routed expert, before their full checkpoints are deleted (optimizer_stats.py). One array task per run, one node
# each, no GPU used. Reads the four checkpoints only; writes only $OUT_ROOT/<tag>.
#   test (one expert bucket + the fp32 buckets, eps run only, ~15 min):
#     sbatch --partition=debug --time=00:30:00 --array=0 --export=ALL,TEST=1 \
#         --output=$SCRATCH/lean/logs/%A_%a-optstats-test.out optimizer_stats.sbatch
#   full:
#     sbatch --output=$SCRATCH/lean/logs/%A_%a-optstats.out optimizer_stats.sbatch
#   report afterwards (login node, stdlib only):
#     python3 $CODE/optimizer_stats.py report h1=$OUT_ROOT/h1 eps=$OUT_ROOT/eps --out $OUT_ROOT/report.md
# Settings (environment): CODE (folder with optimizer_stats.py, default $SCRATCH/lean/optimizer-stats/code),
# OUT_ROOT (default $SCRATCH/lean/optimizer-stats), STEP_A/STEP_B (30/40), WORKERS (buckets in parallel, 6), TEST.
set -euo pipefail
CODE=${CODE:-${SCRATCH}/lean/optimizer-stats/code}
OUT_ROOT=${OUT_ROOT:-${SCRATCH}/lean/optimizer-stats}
R=${SCRATCH}/nemo-rl/results
TAGS=(eps h1)
RUNS=(lean-kimina-v02-abl-eps14-preemptable-20261119 lean-kimina-v02-abl-h1-preemptable-20261118)
i=${SLURM_ARRAY_TASK_ID:-0}
TAG=${TAGS[$i]}
RUN=${RUNS[$i]}
STEP_A=${STEP_A:-30}
STEP_B=${STEP_B:-40}
OUT=${OUT_ROOT}/${TAG}${TEST:+-test}
[[ -f "${CODE}/optimizer_stats.py" ]] || { echo "ERROR: no ${CODE}/optimizer_stats.py" >&2; exit 1; }
for s in "${STEP_A}" "${STEP_B}"; do
    [[ -f "${R}/${RUN}/step_${s}/policy/weights/iter_0000000/.metadata" ]] || { echo "ERROR: no ${RUN}/step_${s}" >&2; exit 1; }
done
[[ -e "${OUT}/groups.jsonl.gz" ]] && { echo "ERROR: ${OUT} already has results" >&2; exit 1; }
mkdir -p "${OUT}"

args=(run --ckpt-b "${R}/${RUN}/step_${STEP_B}" --ckpt-a "${R}/${RUN}/step_${STEP_A}" --out "${OUT}"
      --workers "${WORKERS:-6}" --threads 8 --index-workers 48)
if [[ -n "${TEST:-}" ]]; then
    # the two fp32 buckets (A_log, dt_bias) and expert rank 0, indexing only experts 0-15 and the fp32 tensors
    args+=(--buckets 'dtype_\(torch\.float32|chained_1\.optimizer\.distributed\.dp_group_idx_0\.'
           --experts 0-15 --index-filter '(A_log|dt_bias)$|\.experts\.experts\.')
fi
echo "task ${i}: ${TAG} ${RUN} step ${STEP_A} -> ${STEP_B}, out ${OUT}, test ${TEST:-no}"
date
srun --environment=nemo-rl-apertus2 python3 "${CODE}/optimizer_stats.py" "${args[@]}"
date
ls -la "${OUT}"
