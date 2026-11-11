#!/bin/bash
# Two-node NCCL all-reduce in David's image over Slingshot. Expect "correct=True" and about 30 GB/s busbw.
#   source kisti/env.sh && cd $KISTI_SCRATCH
#   sbatch "${KISTI_SLURM[@]}" -N2 -t 00:20:00 -o nccl_2node_%j.log ~/rl-infra/kisti/nccl_2node.sh
# The image carries its own libfabric with the CXI provider, so no host libraries are mounted
# (mounting KISTI's host libfabric + aws-ofi-nccl, as kisti-docs/pretraining does, gave the same result).
set -euo pipefail
# sbatch runs a spooled copy of this file, so locate the repository by RL_INFRA, not by the script path.
HERE="${RL_INFRA:-${HOME}/rl-infra}/kisti"
source "${HERE}/env.sh"

# KISTI: LL128 gives wrong inter-node all-reduce results on HANGANG. Mandatory on every multi-node job.
export NCCL_PROTO='^LL128' FI_PROVIDER=cxi NCCL_IB_DISABLE=1
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)" MASTER_PORT=29511
srun --network=disable_rdzv_get --ntasks-per-node=4 --container-image="${KISTI_IMAGE}" \
    --container-mounts="${KISTI_SCRATCH}:${KISTI_SCRATCH},${HERE}:${HERE}" --export=ALL \
    bash -c 'export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS; python3 '"${HERE}"'/allreduce.py'
