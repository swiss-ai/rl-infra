#!/bin/bash
#SBATCH --job-name=ctx-podcap
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=pytorch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --propagate=NONE
#SBATCH --output=%x-%j.out

# OpenSandbox pod capacity test (pod_capacity.py): ramps to LEVELS pods (default 64,128,192,256), measures start time
# and a CPU task in every pod at each level, and deletes its own pods at the end. Runs on a compute node, as a run's
# code_gen server does, with the node's own python3 (only requests is needed). Submit from a folder on /scratch.
set -uo pipefail
D=${TOOLS_DIR:?submit with sbatch --export=ALL,TOOLS_DIR=<rl-infra>/shared/scripts/sandbox}
SCRATCH=${SCRATCH:-/scratch/${USER}}
W=${SCRATCH}/long-context
IMAGE=${IMAGE:-/scratch/hanint02/images/nemo-rl-apertus2-cfcfede4-ucclfix.sqsh}
LEVELS=${LEVELS:-64,128,192,256}
mkdir -p "$W/podcap"
echo "$(date +%T) job $SLURM_JOB_ID on $SLURM_JOB_NODELIST: levels $LEVELS"
srun --ntasks=1 --cpu-bind=none --cpus-per-task=64 \
  --container-image="$IMAGE" --container-writable --no-container-mount-home \
  --container-mounts=${SCRATCH}:${SCRATCH},${HOME}:${HOME} \
  /usr/bin/python3.12 -u $D/pod_capacity.py --api-key-file ${OPENSANDBOX_TOKEN_FILE:-${HOME}/.opensandbox_token} \
  --levels "$LEVELS" --sustain-seconds "${SUSTAIN:-0}" --out "$W/podcap/podcap-$SLURM_JOB_ID.json"
echo "$(date +%T) exit $?"
