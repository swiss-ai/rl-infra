#!/bin/bash
# The probe's per-node watcher: ray.sub starts NUMA_WATCH on every node with its output in
# <job>-logs/numa-hbm-<node>.log. Next to that log it writes
#   gpumem-<node>.csv   every PROBE_GPU_INTERVAL_MS (default 2000): nvidia-smi's timestamp, GPU index, memory used and
#                       total (MiB) and utilization (%) of every GPU; all processes on the GPU count (trainer + vLLM)
#   hostmem-<node>.tsv  every PROBE_HOST_INTERVAL_S (default 5): epoch, MemTotal, MemAvailable, MemFree, Shmem and
#                       the free memory of each CPU NUMA node (GiB)
# then becomes numa_watch.sh (HBM spills, hang_watch.sh, Xid messages, Ray log tails), as without the probe.
OUT_DIR="$(dirname "$(readlink -f /proc/$$/fd/1)")"
node="$(hostname -s)"
if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
        -lms "${PROBE_GPU_INTERVAL_MS:-2000}" >> "${OUT_DIR}/gpumem-${node}.csv" 2>&1 &
else
    echo "no nvidia-smi on ${node}" > "${OUT_DIR}/gpumem-${node}.csv"
fi
cpu_nodes=()
for n in /sys/devices/system/node/node*; do
    [[ -n "$(cat "$n/cpulist")" ]] && cpu_nodes+=("$n")
done
( while true; do
    line="$(date +%s)$(awk '/^(MemTotal|MemAvailable|MemFree|Shmem):/ {printf "\t%s=%.1f", $1, $2 / 1048576}' /proc/meminfo)"
    for n in "${cpu_nodes[@]}"; do
        line+="$(awk -v n="${n##*node}" '/MemFree/ {printf "\tcpufree%s=%.1f", n, $4 / 1048576}' "$n/meminfo")"
    done
    echo "${line//:=/=}"
    sleep "${PROBE_HOST_INTERVAL_S:-5}"
  done ) >> "${OUT_DIR}/hostmem-${node}.tsv" 2>&1 &
exec bash "$(dirname "$0")/../../../numa_watch.sh"
