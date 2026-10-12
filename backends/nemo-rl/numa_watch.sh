#!/bin/bash
# Logs, once a minute, how much HOST memory (page cache, anon, shmem) sits on each
# GH200 GPU's HBM NUMA node. On Alps the HBM of GPU i is exposed as a memory-only NUMA
# node (4, 12, 20, 28); host allocations that land there eat GPU memory without showing
# up under any process in nvidia-smi. Started by ray.sub's dispatch script when
# NUMA_WATCH points here. With the mem-bind in launch.sh, file/anon/shmem should stay ~0; "used"
# also counts CUDA's own GPU allocations, so it is not a spill signal by itself.
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
  sleep 60
done
