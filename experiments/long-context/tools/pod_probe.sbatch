#!/bin/bash
#SBATCH --job-name=ctx-podprobe
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=pytorch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:12:00
#SBATCH --propagate=NONE
#SBATCH --output=%x-%j.out

# OpenSandbox health probe (pod_probe.py): a few pods, commands of growing length in each, to see whether the
# service cuts commands off. Runs on a compute node in the run's container, as a run's code_gen server does.
# Submit from a folder on /scratch. PODS (default 3) and SLEEPS (default 1,5,15,30,60,120) pass through.
set -uo pipefail
D=${TOOLS_DIR:?submit with sbatch --export=ALL,TOOLS_DIR=<rl-infra>/experiments/long-context/tools}
SCRATCH=${SCRATCH:-/scratch/${USER}}
IMAGE=${IMAGE:-/scratch/hanint02/images/nemo-rl-apertus2-cfcfede4-ucclfix.sqsh}
echo "$(date +%T) job $SLURM_JOB_ID on $SLURM_JOB_NODELIST"
srun --ntasks=1 --cpu-bind=none --cpus-per-task=8 \
  --container-image="$IMAGE" --container-writable --no-container-mount-home \
  --container-mounts=${SCRATCH}:${SCRATCH},${HOME}:${HOME} \
  /usr/bin/python3.12 -u $D/pod_probe.py --api-key-file ${OPENSANDBOX_TOKEN_FILE:-${HOME}/.opensandbox_token} \
  --pods "${PODS:-3}" --sleeps "${SLEEPS:-1,5,15,30,60,120}"
echo "$(date +%T) exit $?"
