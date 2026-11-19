#!/bin/bash
# eval/run.sh of lean__lean-kimina-v02-abl-h1-20261118, in the shape the handover page asks for.
#
#   MODEL=<HF checkpoint dir> bash eval/run.sh          (from a Clariden login node; takes ~45-60 min once running)
#
# Run it unchanged on `start`, on `ckpt` and on any merge or distilled model; only MODEL changes.
#
# What it measures: Lean 4 whole-proof completion on corrected miniF2F (valid 244 + test 242), each statement twice,
# scored exactly as the RL run validated: rl-infra's NeMo-RL pipeline loads MODEL, generates with its vLLM (fp8, as in
# training), Gym's math_formal_lean builds header + statement + the model's proof and the Kimina Lean server checks it
# (Lean v4.34.0, Mathlib v4.34.0 = the shared install below, 120 s per proof). Reward 1 = proof accepted (no sorry).
# The run validates once at start and stops (grpo.stop_at_validation_metric=accuracy, threshold 0); nothing is
# trained or saved. It needs 12 GH200 nodes (8 for the Megatron policy that refits vLLM, 4 for vLLM), like the run.
#   results.md row:      miniF2F-test (held-out), pass@1 over 2 samples
#   dev benchmark:       miniF2F-valid, pass@1 over 2 samples (used to pick checkpoints; not in results.md)
#
# Needs: read access to infra01's capstor store (image, Lean install, this folder). Settings via env:
#   PARTITION (default preemptable; `normal` cannot be preempted), JOB_TIME (default 01:30:00), NAME (default from MODEL).
set -euo pipefail
: "${MODEL:?MODEL=<HF checkpoint dir>}"
MODEL="$(readlink -f "${MODEL}")"
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
NAME="${NAME:-$(basename "$(dirname "${MODEL}")")-$(basename "${MODEL}")}"
[[ -f "${MODEL}/config.json" && -f "${MODEL}/model.safetensors.index.json" ]] || { echo "No HF checkpoint in ${MODEL}" >&2; exit 1; }

# ---- Pinned code and settings ------------------------------------------------
RL_INFRA_REPO=https://github.com/swiss-ai/rl-infra.git
RL_INFRA_COMMIT=302695a6f372267b6c20d6d4893e3e00bd0a5b26   # branch lean-kimina-verifier; pins NeMo-RL, Gym, Kimina, image
APERTUS_COMMON_COMMIT=494c2be79080aa8fbcd46409b96560f72707cd69  # private repo: bundled next to this script
LEAN=/capstor/store/cscs/swissai/infra01/lean_apertus_v2/lean-v4.34.0   # Lean v4.34.0 + Mathlib 5ed29652 (= the run's)
VAL="${EVAL_DIR}/minif2f-corrected-x2.v434.jsonl"   # 972 rows: corrected miniF2F valid 244 + test 242, twice
SETTINGS=(
    grpo.val_at_start=true grpo.stop_at_validation_metric=accuracy grpo.stop_at_validation_threshold=0.0
    grpo.val_at_end=false checkpointing.enabled=false
    policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.max_new_tokens=16000
    env.nemo_gym.math_formal_lean.resources_servers.math_formal_lean.compilation_timeout=120
    grpo.num_prompts_per_step=32
)
export PARTITION="${PARTITION:-preemptable}" JOB_TIME="${JOB_TIME:-01:30:00}"

WORK="/iopsstor/scratch/cscs/${USER}/expert-evals/${NAME}"
mkdir -p "${WORK}"
RL="${WORK}/rl-infra"
[[ -d "${RL}/.git" ]] || git clone -q -b lean-kimina-verifier "${RL_INFRA_REPO}" "${RL}"
git -C "${RL}" fetch -q origin
git -C "${RL}" checkout -q --force "${RL_INFRA_COMMIT}"

# ---- 1. Once per account (each step skips what exists) -------------------------
# site.env: the shared Lean install instead of a private one
cat > "${RL}/site.env" <<EOF
: "\${LEAN_TOOLCHAIN:=${LEAN}/elan/toolchains/leanprover--lean4---v4.34.0}"
: "\${LEAN_PROJECT:=${LEAN}}"
EOF
# apertus-common (private on GitHub; Clariden has no credentials)
AC="/iopsstor/scratch/cscs/${USER}/nemo-rl-apertus2/apertus-common-${APERTUS_COMMON_COMMIT:0:9}"
if [[ ! -d "${AC}/.git" ]]; then
    mkdir -p "$(dirname "${AC}")"
    git clone -q "${EVAL_DIR}/apertus-common-${APERTUS_COMMON_COMMIT:0:7}.bundle" "${AC}"
    git -C "${AC}" checkout -q --detach "${APERTUS_COMMON_COMMIT}"
fi
bash "${RL}/backends/nemo-rl/lean-kimina/setup.sh" > "${WORK}/setup-kimina.log" 2>&1 \
    || { tail -20 "${WORK}/setup-kimina.log"; exit 1; }

# ---- 2. MODEL as the run's start model ----------------------------------------
# rl-infra stages sft_export (relative to $SCRATCH) as $SCRATCH/models/<basename>, skipping files that exist there:
# refuse a staging of a different model under the same name.
STAGED="/iopsstor/scratch/cscs/${USER}/models/$(basename "${MODEL}")"
if [[ -e "${STAGED}" ]]; then
    for f in "${STAGED}"/*.safetensors; do
        [[ -L "${f}" && "$(readlink -f "${f}")" == "${MODEL}/$(basename "${f}")" ]] && continue
        cmp -s "${f}" "${MODEL}/$(basename "${f}")" 2>/dev/null && continue
        echo "ERROR: ${STAGED} holds another model ($(basename "${f}")); move it away first" >&2; exit 1
    done
    cmp -s "${STAGED}/model.safetensors.index.json" "${MODEL}/model.safetensors.index.json" \
        || { echo "ERROR: ${STAGED} holds another model (index differs); move it away first" >&2; exit 1; }
fi
RECIPE="recipes/eval-$(basename "${MODEL}")-12n.yaml"
cat > "${RL}/${RECIPE}" <<EOF
# Written by eval/run.sh of lean__lean-kimina-v02-abl-h1-20261118: the run's recipe on 12 nodes, on MODEL.
defaults: apertus2-chonk-sftv02-18315-vllm-adamw-code.yaml
sft_export: $(realpath --relative-to="/iopsstor/scratch/cscs/${USER}" "${MODEL}")
cluster:
  num_nodes: 12
policy:
  model_name: \${oc.env:SCRATCH}/models/$(basename "${MODEL}")
  tokenizer:
    name: \${oc.env:SCRATCH}/models/$(basename "${MODEL}")
  generation:
    colocated:
      resources:
        num_nodes: 4
EOF

# ---- 3. Data: the run's training rows (unused, but the pipeline loads them) and the validation file --------------
LEAN_DATA="/iopsstor/scratch/cscs/${USER}/lean"
ROWS="${LEAN_DATA}/data/goedel-train-s1.v434.jsonl"
mkdir -p "${LEAN_DATA}/data"
[[ -s "${ROWS}" ]] || { gunzip -c "${RL}/backends/nemo-rl/lean-kimina/data/goedel-train-s1.v434.jsonl.gz" > "${ROWS}.part" && mv "${ROWS}.part" "${ROWS}"; }
echo "47e7b60208716969386980210fa0530a30b17588d7f44690bb8e75f69e7e4f40  ${ROWS}" | sha256sum -c --quiet
cp -f "${VAL}" "${LEAN_DATA}/data/$(basename "${VAL}")"   # also as Kimina's header rows: the run's header

# ---- 4. Validate once ----------------------------------------------------------
RUN="lean-eval-${NAME}-$(date +%Y%m%d-%H%M%S)"
cd "${RL}"
out="$(RUN="${RUN}" RECIPE="${RECIPE}" HEADER_ROWS="${LEAN_DATA}/data/$(basename "${VAL}")" \
    bash backends/nemo-rl/lean-kimina/train.sh data.train.data_path="${ROWS}" \
    data.validation.data_path="${LEAN_DATA}/data/$(basename "${VAL}")" "${SETTINGS[@]}" 2>&1 | tee /dev/stderr)"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
JOB="$(grep -oE 'Submitted batch job [0-9]+' <<< "${out}" | tail -1 | awk '{print $4}')"
[[ -n "${JOB}" ]] || { echo "ERROR: no job submitted" >&2; exit 1; }
echo "job ${JOB}; waiting for it to end (driver log: /iopsstor/scratch/cscs/${USER}/nemo-rl/slurm/${JOB}-logs/ray-driver.log)"
while [[ -n "$(squeue -h -j "${JOB}" -o %i 2>/dev/null)" ]]; do sleep 60; done
LOG="$(ls /iopsstor/scratch/cscs/${USER}/nemo-rl/logs/${RUN}/exp_*/val_data_step0.jsonl 2>/dev/null | head -1)"
[[ -s "${LOG}" ]] || { echo "ERROR: no validation output for ${RUN}; see the driver log" >&2; exit 1; }

# ---- 5. Score ------------------------------------------------------------------
python3 "${EVAL_DIR}/score.py" "${LOG}" "${VAL}" | tee "${WORK}/scores-${RUN}.txt"
