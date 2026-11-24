#!/bin/bash
# Stage 1 of the four-node design, host side: one GH200 node, no model. Runs host_bench.py's measurements one after
# the other, in the NeMo-RL training container (the torch that Megatron's CPU Adam uses), and merges them into
# $OUT/host.json. Writes only $OUT (results) and $OUT/data (test files, deleted at the end). A requeued job skips
# the measurements that already have results. submit.sh submits it; settings (environment):
#   STAGE1    folder with code/ and out/ (default $SCRATCH/lean/fournode-stage1)
#   IO_GB     bytes per io run (default 96 GB), STREAM_PARAMS_M per socket for stream-adam (default 4000 = 48 GB)
#   IO_BLOCK_MB (16), C2C_GB (4), GRADSUM_PARAMS_M (2000), ADAM_PARAMS_M (2000), STREAM_CHUNK_M (128), START_DELAY (45 s): sizes
#   SKIP      measurements to leave out, e.g. "c2c gradsum"
#SBATCH --job-name=fournode-s1-host
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --time=01:00:00
set -uo pipefail
STAGE1=${STAGE1:-${SCRATCH}/lean/fournode-stage1}
CODE=${STAGE1}/code
OUT=${STAGE1}/out/${SLURM_JOB_ID}-host
DATA=${OUT}/data
IO_GB=${IO_GB:-96}
IO_BLOCK_MB=${IO_BLOCK_MB:-16}
STREAM_PARAMS_M=${STREAM_PARAMS_M:-4000}
STREAM_CHUNK_M=${STREAM_CHUNK_M:-128}
C2C_GB=${C2C_GB:-4}
GRADSUM_PARAMS_M=${GRADSUM_PARAMS_M:-2000}
ADAM_PARAMS_M=${ADAM_PARAMS_M:-2000}
START_DELAY=${START_DELAY:-45}
ENVIRONMENT=${ENVIRONMENT:-nemo-rl-apertus2}
case "${OUT}" in "${SCRATCH}"/lean/fournode-stage1/out/*) ;; *) echo "ERROR: OUT ${OUT} outside the stage-1 folder" >&2; exit 1 ;; esac
mkdir -p "${OUT}"
say() { echo "[host_job $(date '+%F %T')] $*"; }
say "node $(hostname), restart ${SLURM_RESTART_COUNT:-0}, out ${OUT}"
rm -rf "${DATA}"      # test files of an interrupted attempt
mkdir -p "${DATA}"

# The node's layout, outside the container.
{ numactl -H 2>&1 | head -30; nvidia-smi topo -m 2>&1; free -g; df -h /tmp /dev/shm 2>&1; } > "${OUT}/node.txt"

B=(python3 -u "${CODE}/host_bench.py" --out "${OUT}")
# One task per GPU and Grace socket: CPUs bound to the task's NUMA node (map_ldom), so its first-touch memory lands
# there; CUDA_VISIBLE_DEVICES picks the GPU on that socket. Falls back to unbound tasks if Slurm refuses the binding.
per_socket() {
    local name=$1; shift
    local start=$(( $(date +%s) + START_DELAY ))     # container start-up; processes then meet at this time
    local cmd="export CUDA_VISIBLE_DEVICES=\${SLURM_LOCALID} BENCH_RANK=\${SLURM_LOCALID} BENCH_NUMA=\${SLURM_LOCALID}; exec ${B[*]} --start-at ${start} $*"
    srun --ntasks=4 --ntasks-per-node=4 --cpus-per-task=72 --cpu-bind=verbose,map_ldom:0,1,2,3 \
        --environment="${ENVIRONMENT}" bash -c "${cmd}" && return 0
    say "${name}: srun with NUMA binding failed; once more without it"
    start=$(( $(date +%s) + START_DELAY ))
    cmd="export CUDA_VISIBLE_DEVICES=\${SLURM_LOCALID} BENCH_RANK=\${SLURM_LOCALID}; exec ${B[*]} --start-at ${start} $*"
    srun --ntasks=4 --ntasks-per-node=4 --cpus-per-task=72 --environment="${ENVIRONMENT}" bash -c "${cmd}"
}
one_task() {
    srun --ntasks=1 --cpus-per-task=288 --environment="${ENVIRONMENT}" "${B[@]}" "$@"
}
want() { [[ " ${SKIP:-} " != *" $1 "* ]] && [[ ! -e "${OUT}/done.$1" ]]; }
step() {  # step NAME command...: runs it unless skipped or done, marks it done on success
    local name=$1; shift
    want "${name}" || { say "${name}: skipped (done or SKIP)"; return 0; }
    say "${name}: start"
    local t0=${SECONDS}
    if "$@"; then touch "${OUT}/done.${name}"; say "${name}: done in $(( SECONDS - t0 )) s"
    else say "${name}: FAILED (exit $?) after $(( SECONDS - t0 )) s"; fi
}

step facts one_task facts
step c2c per_socket c2c c2c --gb "${C2C_GB}" --reps 5
step gradsum per_socket gradsum gradsum --params-m "${GRADSUM_PARAMS_M}" --chunk-mb 256 --reps 3 --slot 60
step adam per_socket adam adam --params-m "${ADAM_PARAMS_M}" --reps 3 --variants foreach,fused
# Scratch with its default layout (1 stripe of 1 MiB per file), then 4 stripes of 16 MiB per file.
step io-default one_task io --dir "${DATA}/default" --tag default --streams 8,16,32 --total-gb "${IO_GB}" --block-mb "${IO_BLOCK_MB}" --mixed
io_striped() {
    mkdir -p "${DATA}/c4" && lfs setstripe -c 4 -S 16M "${DATA}/c4" || return 1
    one_task io --dir "${DATA}/c4" --tag c4-16m --streams 8,16,32 --total-gb "${IO_GB}" --block-mb "${IO_BLOCK_MB}" --mixed
}
step io-c4 io_striped
# The streamed optimizer step: 4 sockets x STREAM_PARAMS_M parameters (master, m, v: 12 bytes each) on scratch.
step stream-adam per_socket stream-adam stream-adam --dir "${DATA}/stream" --params-m "${STREAM_PARAMS_M}" \
    --chunk-m "${STREAM_CHUNK_M}" --buffers 6 --readers 3 --writers 3 --steps 2 --fused

one_task merge
rm -rf "${DATA}"
say "results in ${OUT}/host.json"
ls -la "${OUT}"
