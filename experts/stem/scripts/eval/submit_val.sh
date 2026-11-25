#!/bin/bash
# bash submit_val.sh <HF export dir> <step> <name> [gen nodes, default 8]
# One evaluation job with vLLM only (run_val.py): <gen nodes> vLLM nodes plus the judge node when the open-QA set is
# included. Container, mounts and environment come from a dry run of run.sh stem (run 5's mix) and the
# NeMo-RL clone of setup.sh (patches 0045 and 0046 load the export into vLLM), so the job sees exactly what run 5 at 32K saw. Ray on Slingshot.
#   SETS=aime25,math500,...  subset of run_val.py's sets (default all)     LIMIT=N  first N rows of every set (debug)
#   THINKING=high (default, the user's rule since 2026-11-19) or medium: the system frame of every set; the level is
#            appended to the evaluation's name.
#   AFTER_CONVERT=1  the export is still being written by a conversion job that SBATCH_MORE's dependency waits for.
#   JOB_TIME (default 03:00:00), SBATCH_MORE pass through.
# Log: ${SCRATCH}/nemo-rl/slurm/<job>-logs/ray-driver.log   Results: ${SCRATCH}/nemo-rl/logs/eval-run5-32k/
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
SCRATCH=${SCRATCH:-/scratch/${USER}}
CLONE=${NEMO_RL_DIR:-${PROJECTS_DIR:-${HOME}/projects}/nemo-rl-apertus2}
grep -q update_weights_from_hf_checkpoint "${CLONE}/nemo_rl/models/generation/vllm/vllm_backend.py" || {
    echo "ERROR: ${CLONE} cannot load an export into vLLM (NeMo-RL patch 0046): rerun setup.sh" >&2; exit 1; }
MODEL=${1:?HF export dir}; STEP=${2:?step}; NAME=${3:?name}; GEN=${4:-8}
[[ -f "${MODEL}/config.json" || "${AFTER_CONVERT:-0}" == 1 ]] || {
    echo "ERROR: ${MODEL} is not an HF export (no config.json); AFTER_CONVERT=1 if a conversion job writes it" >&2; exit 1; }
SETS=${SETS:-aime25,aime26,math500,mmlu_pro,reasoning_gym,enigmata,scienceworld,openqa}
JUDGE=0; [[ ",${SETS}," == *",openqa,"* ]] && JUDGE=1
OUT=${SCRATCH}/long-context
DRY=${OUT}/dry-val-${NAME}.txt
env DEBUG=1 NEMO_RL_DIR=${CLONE} RUN=val-dry DRY_RUN=1 bash ${HERE}/../../../run.sh stem > "${DRY}" 2>&1
grep -q "DRY_RUN: site" "${DRY}" || { echo "ERROR: the dry run printed no settings, see ${DRY}" >&2; exit 1; }
while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"
    [[ "${line}" =~ ^([A-Za-z_][A-Za-z_0-9]*)=(.*)$ ]] || continue
    export "${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
done < <(sed -n '/DRY_RUN: site/,$p' "${DRY}" | tail -n +3)
grep -q "${CLONE}:/opt/nemo-rl" <<< "${MOUNTS}" || { echo "ERROR: the dry run does not mount ${CLONE}" >&2; exit 1; }
EXTRA="--sets ${SETS} --thinking ${THINKING:-high}"; [[ -n "${LIMIT:-}" ]] && EXTRA+=" --limit ${LIMIT}"
export COMMAND="cd /opt/nemo-rl && RAY_DEDUP_LOGS=0 bash ${ROOT}/backends/nemo-rl/job/driver.sh ${HERE}/run_val.py --model ${MODEL} --step ${STEP} --name ${NAME} --gen-nodes ${GEN} ${EXTRA}"
export WANDB_KEY_FILE=${WANDB_KEY_FILE:-${HOME}/.wandb_api_key}
export RAY_NODE_NAME_SUFFIX=-hsn0
unset NUMA_WATCH
mkdir -p ${OUT} && cd ${SCRATCH}/nemo-rl/slurm
sbatch --nodes=$(( GEN + JUDGE )) --account=${ACCOUNT:-in0138} --partition=intTest --time=${JOB_TIME:-03:00:00} \
    --job-name="nrl-${NAME}" --output="${SCRATCH}/nemo-rl/slurm/%j-${NAME}.out" \
    --gres=gpu:4 --exclusive "--comment=field=nlp;appl=pytorch" --propagate=NONE ${SBATCH_MORE:-} \
    ${CLONE}/ray.sub
