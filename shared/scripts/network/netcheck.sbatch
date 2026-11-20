#!/bin/bash
# Which networks a compute node has, their link speeds, and the TCP throughput between two nodes over each.
# Ray (object store transfers of the training batch) uses the address the hostname resolves to; this job shows
# whether that is the slow network.
# With the default 1 node: interfaces and link speeds only. Submitted with --nodes=2 it also sends GB bytes from
# the first node to the second over each address.
#SBATCH --job-name=ctx-netcheck
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=pytorch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=00:08:00
#SBATCH --output=%x-%j.out
set -uo pipefail
GB=${GB:-4}
OUT=${SCRATCH:-/scratch/${USER}}/long-context/netcheck-${SLURM_JOB_ID}
mkdir -p "${OUT}"

cat > "${OUT}/info.sh" <<'EOF'
echo "== $(hostname): hostname resolves to $(getent hosts "$(hostname)" | awk '{print $1}' | head -n1)"
for i in /sys/class/net/*; do
    n=$(basename "$i"); [[ "$n" == lo ]] && continue
    a=$(ip -4 -o addr show "$n" 2>/dev/null | awk '{print $4}' | tr '\n' ' ')
    echo "$n state=$(cat "$i/operstate" 2>/dev/null) speed_mbit=$(cat "$i/speed" 2>/dev/null) mtu=$(cat "$i/mtu" 2>/dev/null) addr=${a:-none}"
done
EOF
srun --ntasks-per-node=1 --cpus-per-task=1 bash "${OUT}/info.sh"

(( SLURM_JOB_NUM_NODES >= 2 )) || exit 0

cat > "${OUT}/sink.py" <<'EOF'
import socket, sys, time
srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", int(sys.argv[1]))); srv.listen(8)
end = time.time() + float(sys.argv[2]); srv.settimeout(5)
while time.time() < end:
    try:
        c, _ = srv.accept()
    except socket.timeout:
        continue
    c.settimeout(60)
    buf = bytearray(1 << 22)
    try:
        while c.recv_into(buf):
            pass
    except OSError:
        pass
    c.close()
EOF
cat > "${OUT}/send.py" <<'EOF'
import socket, sys, time
host, port, gb = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
chunk = bytes(1 << 22); n = int(gb * (1 << 30) / len(chunk))
try:
    s = socket.create_connection((host, port), timeout=10)
except OSError as e:
    print(f"{host}: connect failed: {e}", flush=True); sys.exit(0)
t = time.time()
for _ in range(n):
    s.sendall(chunk)
s.close(); dt = time.time() - t
print(f"{host}: {n * len(chunk) / 1e6 / dt:.0f} MB/s ({gb} GB in {dt:.1f} s, one TCP stream)", flush=True)
EOF

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
A=${NODES[0]}; B=${NODES[1]}
srun -N1 -n1 -w "${B}" --cpus-per-task=4 python3 "${OUT}/sink.py" 5201 150 &
sleep 8
# addresses of node B: the one its hostname resolves to (what Ray uses) and every other IPv4 address it has
srun -N1 -n1 -w "${B}" --overlap --cpus-per-task=1 bash -c "ip -4 -o addr show | awk '\$2 != \"lo\" {print \$2, \$4}'" > "${OUT}/addrs-b.txt"
echo "== addresses of ${B}:"; cat "${OUT}/addrs-b.txt"
while read -r iface cidr; do
    echo "-- ${A} -> ${B} over ${iface} (${cidr%/*})"
    srun -N1 -n1 -w "${A}" --overlap --cpus-per-task=4 python3 "${OUT}/send.py" "${cidr%/*}" 5201 "${GB}" < /dev/null
done < "${OUT}/addrs-b.txt"
wait
