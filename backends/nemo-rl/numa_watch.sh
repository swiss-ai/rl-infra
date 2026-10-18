#!/bin/bash
# Logs, every NUMA_WATCH_INTERVAL_S seconds (default 10), how much HOST memory (page cache, anon, shmem) sits on each
# GH200 GPU's HBM NUMA node. On Alps the HBM of GPU i is exposed as a memory-only NUMA
# node (4, 12, 20, 28); host allocations that land there eat GPU memory without showing
# up under any process in nvidia-smi. Started by ray.sub's dispatch script when
# NUMA_WATCH points here. With the mem-bind in launch.sh, file/anon/shmem should stay ~0; "used"
# also counts CUDA's own GPU allocations, so it is not a spill signal by itself.
# It also starts hang_watch.sh, which logs next to this script's output file, and copies the
# kernel's NVIDIA messages (Xid errors name the faulting GPU address and fault type) to
# xid-<node>.log every minute; the file says so when the kernel log is not readable.
OUT_DIR="$(dirname "$(readlink -f /proc/$$/fd/1)")"
bash "$(dirname "$0")/hang_watch.sh" > "${OUT_DIR}/hang-watch-$(hostname -s).log" 2>&1 &
( while true; do
    dmesg -T 2>&1 | grep -iE "xid|nvrm|nvidia-uvm|nvidia-nvlink|dmesg:" > "${OUT_DIR}/xid-$(hostname -s).log"
    sleep 60
  done ) &
while true; do
  line="$(date +%T)"
  for n in /sys/devices/system/node/node*; do
    [[ -n "$(cat "$n/cpulist")" ]] && continue  # CPU nodes: skip, only HBM nodes
    line+="$(awk -v n="${n##*node}" '
      /MemTotal/ {t=$4} /MemUsed/ {u=$4} /FilePages/ {f=$4} /AnonPages/ {a=$4} /Shmem:/ {s=$4}
      END { if (t > 0) printf " | node%s used=%.1fG file=%.1fG anon=%.1fG shmem=%.1fG", n, u/1048576, f/1048576, a/1048576, s/1048576 }
    ' "$n/meminfo")"
  done
  echo "$line"
  sleep "${NUMA_WATCH_INTERVAL_S:-10}"
done
