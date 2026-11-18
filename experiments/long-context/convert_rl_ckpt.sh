#!/bin/bash
# bash convert_rl_ckpt.sh <run name> <step> <export name>
#   e.g. bash convert_rl_ckpt.sh sftv02-stem-mix-run4-33n 100 sftv02-stem-run4-step100
# Converts the policy weights of a NeMo-RL checkpoint (/scratch/$USER/nemo-rl/results/<run>/step_<step>) to an HF
# export in /scratch/$USER/chonk-sft/models/<export name>, with hanint05's conversion job (one node, a few minutes)
# and the reference config our SFT exports use. The checkpoint is only read.
# Converter code: hanint05's own checkouts, read in place (ours under chonk-sft/src were removed on 2026-11-16);
# the job checks that Megatron-Bridge and Megatron-LM are at the commits recorded in our SFT exports
# (hfconverter acf1dbe, Bridge 6ccdb3c, Megatron-LM-MoE-bridge 42e7a63). git refuses another user's repository
# unless it is marked safe: GIT_CONFIG_* does that for this job only, without touching any config file.
set -euo pipefail
run="${1:?usage: convert_rl_ckpt.sh <run name> <step> <export name>}"
step="${2:?step}"
name="${3:?export name}"
ROOT="${ROOT:-/scratch/${USER}/chonk-sft}"
SRC="${SRC:-/scratch/hanint05}"
CKPT="/scratch/${USER}/nemo-rl/results/${run}/step_${step}/policy/weights/iter_0000000"
JOB="${JOB:-/scratch/hanint05/hf-exports/bridge/convert_bridge_kisti.sbatch}"
IMAGE="${IMAGE:-/scratch/hanint02/images/nemo-rl-apertus2-cfcfede4-ucclfix.sqsh}"
[[ -f "${JOB}" ]] || { echo "ERROR: no job script at ${JOB}" >&2; exit 1; }
[[ -f "${CKPT}/common.pt" ]] || { echo "ERROR: no checkpoint at ${CKPT}" >&2; exit 1; }
output="${ROOT}/models/${name}"
[[ ! -e "${output}" ]] || { echo "ERROR: ${output} exists already" >&2; exit 1; }
cd "${ROOT}"   # the job's log path is relative, and jobs must start on /scratch
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
sbatch --propagate=NONE --job-name=ctx-convert --export="ALL,CHECKPOINT=${CKPT},HF_OUTPUT=${output},IMAGE=${IMAGE},\
HFCONVERTER_ROOT=${SRC}/hfconverter,BRIDGE_ROOT=${SRC}/Megatron-Bridge,\
MEGATRON_ROOT=${SRC}/Megatron-LM-MoE-bridge,HF_REFERENCE=${ROOT}/reference-chonk-instruct" "${JOB}"
echo "export will be: ${output}"
echo "job log: ${ROOT}/logs/ctx-convert-<job>.out"
