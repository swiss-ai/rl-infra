#!/bin/bash
# Logs, every NUMA_WATCH_INTERVAL_S seconds (default 10), how much HOST memory (page cache, anon, shmem) sits on each
# GH200 GPU's HBM NUMA node. On Alps the HBM of GPU i is exposed as a memory-only NUMA
# node (4, 12, 20, 28); host allocations that land there eat GPU memory without showing
# up under any process in nvidia-smi. Started by ray.sub's dispatch script when
# NUMA_WATCH points here. With the mem-bind in launch.sh, file/anon/shmem should stay ~0; "used"
# also counts CUDA's own GPU allocations, so it is not a spill signal by itself.
# Each line also gives the free memory of the CPU NUMA nodes (cpufree<N>), to tell a spill
# caused by full CPU memory from one that ignores the binding. When host memory on any HBM
# node exceeds NUMA_WATCH_PROC_GB (default 2), at most once per NUMA_WATCH_PROC_S seconds
# (default 60) it logs, for each of the user's processes holding over 0.25 GB there, the
# amount per HBM node, the memory policy of those pages (from /proc/<pid>/numa_maps: bind,
# prefer, default) and the four largest kinds (anon, heap, or the mapped file, e.g.
# /dev/nvidia* for CUDA pinned or managed memory), with the command line (Ray actors show as
# ray::<Class>.<method>). Reading numa_maps walks the page tables, which is why it only runs
# once a spill is visible.
# It also starts hang_watch.sh, which logs next to this script's output file, and copies the
# kernel's NVIDIA messages (Xid errors name the faulting GPU address and fault type) to
# xid-<node>.log every minute; the file says so when the kernel log is not readable.
# Every NUMA_WATCH_RAYLOG_S seconds (default 60) and when the job ends it copies the tail of
# this node's Ray logs (raylet, worker stderr and core-worker logs, which live in the
# node-local /tmp/ray and are otherwise lost; this script is usually killed with Ray before
# it sees the job end) to raylogs/<node>/ next to its output file.
OUT_DIR="$(dirname "$(readlink -f /proc/$$/fd/1)")"
bash "$(dirname "$0")/hang_watch.sh" > "${OUT_DIR}/hang-watch-$(hostname -s).log" 2>&1 &
( while true; do
    dmesg -T 2>&1 | grep -iE "xid|nvrm|nvidia-uvm|nvidia-nvlink|dmesg:" > "${OUT_DIR}/xid-$(hostname -s).log"
    sleep 60
  done ) &

hbm_nodes=""
for n in /sys/devices/system/node/node*; do
  [[ -z "$(cat "$n/cpulist")" ]] && hbm_nodes+="${hbm_nodes:+|}${n##*node}"
done

proc_report() {
  for p in /proc/[0-9]*; do
    [[ -O "$p" && -r "$p/numa_maps" ]] || continue
    awk -v hbm="^(${hbm_nodes})$" -v pid="${p#/proc/}" -v cmd="$(tr '\0' ' ' 2>/dev/null < "$p/cmdline" | cut -c1-100)" '
      {
        ps = 4
        for (i = 3; i <= NF; i++) if ($i ~ /^kernelpagesize_kB=/) ps = substr($i, 19) + 0
        kind = "other"
        if ($0 ~ / anon=/) kind = "anon"
        if ($0 ~ / heap/) kind = "heap"
        if (match($0, / file=[^ ]+/)) kind = substr($0, RSTART + 6, RLENGTH - 6)
        for (i = 3; i <= NF; i++) if ($i ~ /^N[0-9]+=/) {
          split(substr($i, 2), a, "=")
          if (a[1] ~ hbm) { kb = a[2] * ps; tot += kb; node[a[1]] += kb; pol[$2] += kb; kd[kind] += kb }
        }
      }
      END {
        if (tot < 262144) exit
        printf "  pid %s %.1fG on HBM:", pid, tot / 1048576
        for (k in node) printf " N%s=%.1fG", k, node[k] / 1048576
        printf " | policy"; for (k in pol) printf " %s=%.1fG", k, pol[k] / 1048576
        printf " | kind"
        for (j = 0; j < 4; j++) {
          best = ""; for (k in kd) if (best == "" || kd[k] > kd[best]) best = k
          if (best == "") break
          printf " %s=%.1fG", best, kd[best] / 1048576; delete kd[best]
        }
        printf " | %s\n", cmd
      }' "$p/numa_maps" 2>/dev/null
  done
}

save_ray_logs() {
  local src dst="${OUT_DIR}/raylogs/$(hostname -s)"
  src="$(ls -d /tmp/ray/session_[0-9]*/logs 2>/dev/null | tail -1)"
  [[ -n "${src}" ]] || return
  mkdir -p "${dst}"
  tail -n 3000 "${src}/raylet.out" > "${dst}/raylet.out" 2>/dev/null
  for f in "${src}"/*.err "${src}"/python-core-worker-*.log; do
    [[ -s "${f}" ]] && tail -n 300 "${f}" > "${dst}/$(basename "${f}")"
  done
}
trap 'save_ray_logs; exit 0' TERM INT

last_report=0
last_raylogs=0
while [[ ! -e "${OUT_DIR}/ENDED" ]]; do
  line="$(date +%T)"
  spill_kb=0
  for n in /sys/devices/system/node/node*; do
    if [[ -n "$(cat "$n/cpulist")" ]]; then
      line+="$(awk -v n="${n##*node}" '/MemFree/ {printf " cpufree%s=%.0fG", n, $4/1048576}' "$n/meminfo")"
      continue
    fi
    line+="$(awk -v n="${n##*node}" '
      /MemTotal/ {t=$4} /MemUsed/ {u=$4} /FilePages/ {f=$4} /AnonPages/ {a=$4} /Shmem:/ {s=$4}
      END { if (t > 0) printf " | node%s used=%.1fG file=%.1fG anon=%.1fG shmem=%.1fG", n, u/1048576, f/1048576, a/1048576, s/1048576 }
    ' "$n/meminfo")"
    kb="$(awk '/FilePages/ {f=$4} /AnonPages/ {a=$4} END {print f+a}' "$n/meminfo")"
    (( kb > spill_kb )) && spill_kb=$kb
  done
  echo "$line"
  now=$(date +%s)
  if (( spill_kb > ${NUMA_WATCH_PROC_GB:-2} * 1048576 && now - last_report >= ${NUMA_WATCH_PROC_S:-60} )); then
    echo "$(date +%T) host memory on HBM above ${NUMA_WATCH_PROC_GB:-2}G, per process:"
    proc_report
    last_report=$now
  fi
  if (( now - last_raylogs >= ${NUMA_WATCH_RAYLOG_S:-60} )); then
    save_ray_logs
    last_raylogs=$now
  fi
  sleep "${NUMA_WATCH_INTERVAL_S:-10}"
done
save_ray_logs
