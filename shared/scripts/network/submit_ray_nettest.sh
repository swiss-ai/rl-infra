#!/bin/bash
# bash submit_ray_nettest.sh [node name suffix]
# A 2-node Ray cluster started by ray.sub exactly as an RL run starts it (same image, mounts and container
# environment, taken from a dry run of launch-code-ctx.sh), with ray_nettest.py as the driver command: it reports the
# address Ray uses on each node and the speed of an object transfer between the nodes.
#   suffix  empty: Ray on the address the node name resolves to (the 1 Gbit management Ethernet).
#           -hsn0: RAY_NODE_NAME_SUFFIX=-hsn0, Ray on the first Slingshot interface.
# Output: ${SCRATCH}/long-context/raynet/<job>-logs/ray-driver.log (lines starting with RESULT).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH=${SCRATCH:-/scratch/${USER}}
SUFFIX=${1:-}
OUT=${SCRATCH}/long-context/raynet
mkdir -p "${OUT}"
DRY=${OUT}/dry-run.txt
DRY_RUN=1 NODES=16 DEBUG=1 bash ${HERE}/../../../experts/code/scripts/long-context/launch-code-ctx.sh 32768 > "${DRY}" 2>&1
grep -q "DRY_RUN: site" "${DRY}" || { echo "ERROR: the dry run printed no settings, see ${DRY}" >&2; exit 1; }
while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"
    [[ "${line}" =~ ^([A-Za-z_][A-Za-z_0-9]*)=(.*)$ ]] || continue
    export "${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
done < <(sed -n '/DRY_RUN: site/,$p' "${DRY}" | tail -n +3)
export BASE_LOG_DIR=${OUT}
export COMMAND="/usr/bin/python3.12 ${HERE}/ray_nettest.py"
unset WANDB_API_KEY NUMA_WATCH
if [[ -n "${SUFFIX}" ]]; then export RAY_NODE_NAME_SUFFIX=${SUFFIX}; else unset RAY_NODE_NAME_SUFFIX; fi
cd "${OUT}"
sbatch --nodes=2 --account=${ACCOUNT:-in0138} --partition=intTest --time=00:20:00 \
    --job-name="ctx-raynet${SUFFIX}" --output="${OUT}/%j-raynet${SUFFIX}.out" \
    --gres=gpu:4 --exclusive "--comment=field=nlp;appl=pytorch" --propagate=NONE \
    ${NEMO_RL_DIR:-${PROJECTS_DIR:-${HOME}/projects}/nemo-rl-apertus2}/ray.sub
