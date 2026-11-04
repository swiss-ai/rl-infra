#!/bin/bash
# Submits the conversion of one Megatron SFT checkpoint to HF (one node, about 5 minutes):
#   RUN=<run's checkpoints> JOB=<export sbatch script> IMAGE=<image> bash convert.sh 8075
# It runs the SFT export job, which takes every path from the environment, on the clean clones from
# setup.sh, so the commits match the conversion that passed parity. The export lands in models/.
set -euo pipefail

iter="$(printf 'iter_%07d' "${1:?usage: convert.sh <iteration>}")"
ROOT="${ROOT:-/scratch/${USER}/chonk-sft}"
RUN="${RUN:?Set RUN to the directory of the SFT checkpoints.}"
JOB="${JOB:?Set JOB to the sbatch script of the HF export.}"
IMAGE="${IMAGE:?Set IMAGE to the NeMo-RL image.}"

[[ -f "${RUN}/${iter}/common.pt" ]] || { echo "ERROR: no checkpoint at ${RUN}/${iter}" >&2; exit 1; }
output="${ROOT}/models/chonk-lc-16k-v0.1-ep1-${iter}"
mkdir -p "${ROOT}/models"

cd "${ROOT}"   # the job's log path is relative, and jobs must start on /scratch
sbatch --export="ALL,CHECKPOINT=${RUN}/${iter},HF_OUTPUT=${output},IMAGE=${IMAGE},\
HFCONVERTER_ROOT=${ROOT}/src/hfconverter,BRIDGE_ROOT=${ROOT}/src/Megatron-Bridge,\
MEGATRON_ROOT=${ROOT}/src/Megatron-LM-MoE-bridge,HF_REFERENCE=${ROOT}/reference-chonk-instruct" "${JOB}"
echo "export: ${output}"
