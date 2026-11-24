#!/bin/bash
# Stage 2: ScratchAdamW inside Megatron's HybridDeviceOptimizer on one GH200 node (hdo_gpu_test.py), in the NeMo-RL
# training container. Reads the pinned hybrid_optimizer.py from the Megatron clone (git show, the clone is not
# changed) and applies apertus2/patches/megatron to a copy in $OUT. Writes only $OUT and $OUT/state (deleted at the
# end). Settings (environment):
#   STAGE2         folder with code/ and out/ (default $SCRATCH/lean/fournode-stage2)
#   MEGATRON_DIR   the Megatron-LM-MoE clone (default ~/projects/megatron-lm-moe-apertus2)
#   SPEED_M        million params for the speed check (default 2000; 0 skips it)
#   CONTAINER_EDF  container environment (default nemo-rl-apertus2; not ENVIRONMENT, which Slurm sets to BATCH)
#SBATCH --job-name=fournode-s2-hdo
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --time=00:30:00
set -euo pipefail
STAGE2=${STAGE2:-${SCRATCH}/lean/fournode-stage2}
CODE=${STAGE2}/code
OUT=${STAGE2}/out/${SLURM_JOB_ID}-hdo
MEGATRON_DIR=${MEGATRON_DIR:-${HOME}/projects/megatron-lm-moe-apertus2}
MEGATRON_COMMIT=0536034901962160aac9742d134a84c55e28d224
SPEED_M=${SPEED_M:-2000}
CONTAINER_EDF=${CONTAINER_EDF:-nemo-rl-apertus2}
case "${OUT}" in "${SCRATCH}"/lean/fournode-stage2/out/*) ;; *) echo "ERROR: OUT ${OUT} outside the stage-2 folder" >&2; exit 1 ;; esac
mkdir -p "${OUT}"
say() { echo "[hdo_job $(date '+%F %T')] $*"; }

# the patched hybrid_optimizer.py, exactly as setup.sh would produce it from the pin
rel=megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py
rm -rf "${OUT}/mlm" && mkdir -p "${OUT}/mlm/$(dirname "${rel}")"
git -C "${MEGATRON_DIR}" show "${MEGATRON_COMMIT}:${rel}" > "${OUT}/mlm/${rel}"
(cd "${OUT}/mlm" && for p in "${CODE}"/megatron-patches/*.patch; do git apply --include="${rel}" "${p}"; done)
grep -q "skip_next_step" "${OUT}/mlm/${rel}" || { say "ERROR: patch 0007 did not reach ${rel}"; exit 1; }
say "node $(hostname), patched ${rel} ($(sha256sum "${OUT}/mlm/${rel}" | cut -c1-12))"

set +e
srun --ntasks=1 --cpus-per-task=288 --environment="${CONTAINER_EDF}" \
    python3 "${CODE}/gpu/hdo_gpu_test.py" --hdo "${OUT}/mlm/${rel}" --root "${OUT}/state" --speed-m "${SPEED_M}" \
    | tee "${OUT}/results.jsonl"
rc=${PIPESTATUS[0]}
set -e
rm -rf "${OUT}/state"
say "exit ${rc}, results in ${OUT}/results.jsonl"
exit "${rc}"
