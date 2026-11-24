#!/bin/bash
# Stage 1 of the four-node design, generation side: Chonk in vLLM on one node (DP4 + expert parallel, as the
# chonk-lean-sampling sampler starts it), with fp8 weights like the GRPO run, the KV cache in KV (auto = bf16, or
# fp8) and sleep mode on. conc_client.py then keeps 64/128/256 GRPO Lean prompts per GPU in flight (continuous
# admission) and measures tokens/s per GPU from vLLM's counters; at the end it times sleep level 1/2, wake-up and
# reload_weights. Writes only $OUT. A requeued job starts over in a new attempt folder.
# submit.sh submits it once per KV type. Settings (environment):
#   KV (auto|fp8), WEIGHTS (fp8|bf16, default fp8), LEVELS (64,128,256), WARM / MEASURE seconds per level (480/420),
#   GPU_SHARE (0.88, the GRPO run's), MNBT (max-num-batched-tokens, 8192), STAGE1 (default $SCRATCH/lean/fournode-stage1),
#   MODEL, DATA, TOKENIZER, SYSTEM_PROMPT_FILE, CHONK_ROOT (container + venv), SAMPLER_DIR (sampler.py)
#SBATCH --job-name=fournode-s1-vllm
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --time=01:20:00
set -uo pipefail
STAGE1=${STAGE1:-${SCRATCH}/lean/fournode-stage1}
KV=${KV:-auto}
WEIGHTS=${WEIGHTS:-fp8}
LEVELS=${LEVELS:-64,128,256}
WARM=${WARM:-480}
MEASURE=${MEASURE:-420}
GPU_SHARE=${GPU_SHARE:-0.88}
MNBT=${MNBT:-8192}
CHONK_ROOT=${CHONK_ROOT:-${SCRATCH}/chonk-proofnet}
SAMPLER_DIR=${SAMPLER_DIR:-${SCRATCH}/lean-sampling}
MODEL=${MODEL:-/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/posttraining/chonk-lc-16k-v0.2-iter_0018315}
DATA=${DATA:-${SCRATCH}/lean/goedel-full-5c99dce/goedel-clean.v434.jsonl}
TOKENIZER=${TOKENIZER:-${CHONK_ROOT}/tokenizer}
SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-${SCRATCH}/rl-infra/envs/apertus2-system.json}
PORT=${PORT:-8000}
OUT=${STAGE1}/out/${SLURM_JOB_ID}-vllm-kv${KV}/r${SLURM_RESTART_COUNT:-0}
case "${OUT}" in "${SCRATCH}"/lean/fournode-stage1/out/*) ;; *) echo "ERROR: OUT ${OUT} outside the stage-1 folder" >&2; exit 1 ;; esac
mkdir -p "${OUT}"
say() { echo "[vllm_job $(date '+%F %T')] $*"; }
for f in "${MODEL}/config.json" "${DATA}" "${TOKENIZER}/tokenizer.json" "${SYSTEM_PROMPT_FILE}" "${CHONK_ROOT}/environment.toml" \
         "${CHONK_ROOT}/venv/bin/python" "${SAMPLER_DIR}/sampler.py" "${STAGE1}/code/conc_client.py"; do
    [[ -e "${f}" ]] || { say "ERROR: missing ${f}"; exit 1; }
done
end=$(SLURM_TIME_FORMAT=%s squeue -h -j "${SLURM_JOB_ID}" -o %e 2> /dev/null | head -n 1)
[[ "${end}" =~ ^[0-9]+$ ]] || end=$(( $(date +%s) + 80 * 60 ))
say "node $(hostname), KV ${KV}, weights ${WEIGHTS}, levels ${LEVELS}, job ends $(date -d "@${end}" '+%T'), out ${OUT}"
max_seqs=$(tr ',' '\n' <<< "${LEVELS}" | sort -n | tail -1)
URL="http://127.0.0.1:${PORT}"
container=(--environment="${CHONK_ROOT}/environment.toml"
  --container-mounts="${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor,/ritom:/ritom")
server=""
stop_server() { [[ -n "${server}" ]] && { kill -TERM "${server}" 2> /dev/null; sleep 20; kill -KILL "${server}" 2> /dev/null; wait "${server}" 2> /dev/null; }; server=""; }
trap 'say "signal: stopping"; stop_server; exit 143' TERM INT

# start_server WEIGHTS SLEEP(1|0) LOG: vLLM as the sampler starts it (1 node), plus the settings under test
start_server() {
    local weights=$1 sleep_mode=$2 logf=$3
    srun --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=288 "${container[@]}" \
        -u bash -s -- "${MODEL}" "${GPU_SHARE}" "${PORT}" "${max_seqs}" "${MNBT}" "${KV}" "${weights}" "${sleep_mode}" \
        > "${logf}" 2>&1 <<'VLLM' &
set -euo pipefail
model=$1; gpu_share=$2; port=$3; max_seqs=$4; mnbt=$5; kv=$6; weights=$7; sleep_mode=$8
export LOCAL_WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3
export XDG_CACHE_HOME="/tmp/vllm-${SLURM_JOB_ID}"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm" FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor" TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/extensions"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_SERVER_DEV_MODE=1      # the /sleep, /wake_up and /collective_rpc endpoints
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
python3 -c 'import vllm, torch; print("versions: vllm", vllm.__version__, "torch", torch.__version__)' || true
args=(
  "$model" --host 0.0.0.0 --port "$port" --served-model-name chonk
  --tensor-parallel-size 1 --data-parallel-size 4 --data-parallel-size-local 4
  --enable-expert-parallel --all2all-backend allgather_reducescatter --moe-backend triton
  --dtype bfloat16 --trust-remote-code --numa-bind
  --load-format runai_streamer
  --model-loader-extra-config '{"distributed":true,"concurrency":4,"memory_limit":17179869184}'
  --max-model-len 16384 --gpu-memory-utilization "$gpu_share"
  --max-num-seqs "$max_seqs" --max-num-batched-tokens "$mnbt"
  --kv-cache-dtype "$kv" --api-server-count 4
)
[[ "$weights" == fp8 ]] && args+=(--quantization fp8)
[[ "$sleep_mode" == 1 ]] && args+=(--enable-sleep-mode)
echo "vllm serve ${args[*]}"
exec vllm serve "${args[@]}"
VLLM
    server=$!
    local t0=${SECONDS}
    until curl -sf --max-time 10 "${URL}/health" > /dev/null; do
        kill -0 "${server}" 2> /dev/null || { say "vLLM (${weights}, sleep ${sleep_mode}) exited after $(( SECONDS - t0 )) s: $(tail -3 "${logf}" | tr '\n' ' ' | cut -c1-400)"; server=""; return 1; }
        (( SECONDS - t0 < 1500 )) || { say "vLLM not ready after 25 min"; stop_server; return 1; }
        sleep 10
    done
    say "vLLM ready after $(( SECONDS - t0 )) s (weights ${weights}, KV ${KV}, sleep mode ${sleep_mode})"
    echo "weights=${weights} sleep_mode=${sleep_mode} ready_s=$(( SECONDS - t0 ))" > "${OUT}/server.txt"
}

started=0
for attempt in "${WEIGHTS} 1" "${WEIGHTS} 0" "bf16 1"; do
    read -r w s <<< "${attempt}"
    log_f="${OUT}/vllm-${w}-sleep${s}.log"
    start_server "${w}" "${s}" "${log_f}" && { started=1; break; }
    stop_server
    (( $(date +%s) + 40 * 60 < end )) || { say "too little time left for another start"; break; }
done
(( started )) || { say "ERROR: vLLM never started"; exit 1; }
sed 's/\x1b\[[0-9;]*m//g' "${log_f}" | grep -E "versions:|KV cache|Maximum concurrency|Available KV|Model loading took|weights took|CUDA graph|sleep" \
    | head -40 >> "${OUT}/server.txt"
read -r w s < <(sed -n 's/^weights=\([a-z0-9]*\) sleep_mode=\([01]\).*/\1 \2/p' "${OUT}/server.txt")

PY=("${CHONK_ROOT}/venv/bin/python" -u "${STAGE1}/code/conc_client.py" --url "${URL}" --out "${OUT}")
say "sweep: levels ${LEVELS} per GPU, ${WARM} s warm-up + ${MEASURE} s measured each"
"${PY[@]}" sweep --levels "${LEVELS}" --warm "${WARM}" --measure "${MEASURE}" --deadline $(( end - 12 * 60 )) \
    --data "${DATA}" --tokenizer "${TOKENIZER}" --system-prompt-file "${SYSTEM_PROMPT_FILE}" --sampler-dir "${SAMPLER_DIR}" \
    || say "sweep ended with $?"
if [[ "${s}" == 1 ]]; then
    "${PY[@]}" sleepwake --drain 120 || say "sleepwake ended with $?"
else
    say "no sleep mode in this server: sleep/wake not measured"
fi
stop_server
say "results in ${OUT}"
ls -la "${OUT}"
