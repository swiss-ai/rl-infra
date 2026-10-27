# KISTI HANGANG settings shared by the scripts in this directory. Source it on the login node:
#   source kisti/env.sh
# Every variable can be set beforehand to override the default.
: "${KISTI_ACCOUNT:=in0138}"
# KISTI dedicated intTest to us (2026-10-27); `normal` rejects our group.
: "${KISTI_PARTITION:=intTest}"
# Mandatory on every job; see `showappl` for the application tags.
: "${KISTI_COMMENT:=field=nlp;appl=misc}"
# Inside a pyxis container $USER is root: source this on the host first (jobs inherit the exports).
: "${KISTI_SCRATCH:=/scratch/${SLURM_JOB_USER:-${USER}}}"
# David's Apertus 2 NeMo-RL image (same file as IMAGE_WORK in backends/nemo-rl/apertus2/env).
: "${KISTI_IMAGE:=${KISTI_SCRATCH}/images/nemo-rl-apertus2-cfcfede4-ucclfix.sqsh}"
: "${KISTI_MODEL:=${KISTI_SCRATCH}/models/apertus2-chonk-lc-iter3576}"
export KISTI_ACCOUNT KISTI_PARTITION KISTI_COMMENT KISTI_SCRATCH KISTI_IMAGE KISTI_MODEL

# sbatch/srun flags every job needs: whole nodes (single-node jobs need --exclusive for the memory) and all 4 GPUs.
KISTI_SLURM=(-A "${KISTI_ACCOUNT}" -p "${KISTI_PARTITION}" --comment="${KISTI_COMMENT}" --exclusive --gres=gpu:4 --mem=0)
