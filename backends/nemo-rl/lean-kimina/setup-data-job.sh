#!/bin/bash
# Body of the data job (setup-data.sh submits it). Writes only under $LEAN_DATA and /dev/shm (deleted at exit).
set -euo pipefail
source "${ROOT}/sites/load.sh" && rl_infra_load "${LEAN_KIMINA_DIR}/env"
source "${LEAN_KIMINA_DIR}/pin"
source "${LEAN_KIMINA_DIR}/kimina.sh"
source "${ROOT}/backends/nemo-rl/apertus2/pin"   # GYM_URL, GYM_COMMIT
LEAN_DIR="${ROOT}/backends/nemo-rl/lean"
PY="${KIMINA_INSTALL}/venv/bin/python"
SRC="${LEAN_DATA}/source/lean.jsonl"
DATA="${LEAN_DATA}/data"
SHM="${KIMINA_SHM:-/dev/shm/${USER}-lean-kimina-data-${SLURM_JOB_ID}}"
WORK="${LEAN_DATA}/work-${SLURM_JOB_ID}"
# Keeps the job's own exit status: under `set -e` a failing command in an EXIT trap would replace it.
cleanup() {
    local rc=$?
    set +e
    [[ -n "${KIMINA_PID:-}" ]] && kill "${KIMINA_PID}" 2>/dev/null && sleep 2
    pkill -u "${USER}" -f "$(kimina_repl_bin)" 2>/dev/null
    rm -rf "${SHM}" "${WORK}"
    exit "${rc}"
}
trap cleanup EXIT
step() { echo "$(date +%T) == $*"; }
mkdir -p "${WORK}" "${DATA}" "${LEAN_DATA}/source" "${LEAN_DATA}/logs"

# Gym's two files at the Gym pin, from GitHub (no NeMo-RL clone needed for this).
GYM="${LEAN_DATA}/gym-${GYM_COMMIT:0:9}"
if [[ ! -s "${GYM}/data/minif2f_test.jsonl" ]]; then
    step "fetching Gym math_formal_lean files at ${GYM_COMMIT:0:9}"
    raw="https://raw.githubusercontent.com/${GYM_URL#https://github.com/}"
    raw="${raw%.git}/${GYM_COMMIT}/resources_servers/math_formal_lean"
    mkdir -p "${GYM}/data.part"
    for f in prepare_nemotron_math_proofs.py app.py; do curl -sSfL "${raw}/${f}" -o "${GYM}/data.part/${f}"; done
    curl -sSfL "${raw}/data/minif2f_test.jsonl" -o "${GYM}/data.part/minif2f_test.jsonl"
    mkdir -p "${GYM}/data"
    mv "${GYM}/data.part/prepare_nemotron_math_proofs.py" "${GYM}/data.part/app.py" "${GYM}/"
    mv "${GYM}/data.part/minif2f_test.jsonl" "${GYM}/data/"
    rmdir "${GYM}/data.part"
fi

# As backends/nemo-rl/lean/setup-data-job.sh, step by step.
URL=https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1/resolve/97229c590831adfe96202f5cd071d444d535bf91/data/lean.jsonl
if [[ ! -s "${SRC}" ]]; then
    step "downloading Nemotron-Math-Proofs-v1 lean.jsonl (29.5 GB, pinned revision)"
    mkdir -p "${LEAN_DATA}/source/parts"
    "${PY}" "${LEAN_DIR}/download.py" "${URL}" 29525155225 "${LEAN_DATA}/source/parts" "${SRC}.part"
    mv "${SRC}.part" "${SRC}" && rm -rf "${LEAN_DATA}/source/parts"
fi
if [[ ! -d "${LEAN_DATA}/source/reduced" ]]; then
    step "reducing it to what Gym's script reads"
    rm -rf "${LEAN_DATA}/source/reduced.part"
    "${PY}" "${LEAN_DIR}/reduce_nemotron.py" "${SRC}" "${LEAN_DATA}/source/reduced.part" "${DATA}/nemotron_sample_proofs.jsonl"
    mv "${LEAN_DATA}/source/reduced.part" "${LEAN_DATA}/source/reduced"
fi
if [[ ! -s "${DATA}/train.jsonl" || ! -s "${DATA}/validation.jsonl" ]]; then
    step "selecting the training theorems with Gym's prepare_nemotron_math_proofs.py"
    # A copy of Gym's script, which writes into data/ next to itself.
    mkdir -p "${WORK}/gym-lean/data" && cp "${GYM}/prepare_nemotron_math_proofs.py" "${WORK}/gym-lean/"
    "${PY}" "${WORK}/gym-lean/prepare_nemotron_math_proofs.py" --dataset-path "${LEAN_DATA}/source/reduced" \
        --output-name nemotron_math_proofs
    "${PY}" - "${WORK}/gym-lean/data/nemotron_math_proofs.jsonl" "${GYM}/data/minif2f_test.jsonl" "${DATA}" <<'PY'
import json, sys
ref = {"type": "responses_api_agents", "name": "math_formal_lean_simple_agent"}
train, valid, data = sys.argv[1:4]
for src, dst in ((train, f"{data}/train.jsonl.part"), (valid, f"{data}/validation.jsonl.part")):
    with open(src) as f, open(dst, "w") as g:
        n = 0
        for line in f:
            row = json.loads(line); row["agent_ref"] = ref; g.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
    print(dst, n)
PY
    "${PY}" "${LEAN_DIR}/fix_rows.py" "${DATA}/train.jsonl.part"
    mv "${DATA}/validation.jsonl.part" "${DATA}/validation.jsonl"
    mv "${DATA}/train.jsonl.part" "${DATA}/train.jsonl"
fi
wc -l "${DATA}/train.jsonl" "${DATA}/validation.jsonl"

# Kimina server in this job: Lean into node memory, REPLs warmed with the rows' import header.
header=$("${PY}" "${LEAN_KIMINA_DIR}/dominant_header.py" "${KIMINA_INSTALL}/kimina-lean-server" "${DATA}/train.jsonl" "${DATA}/validation.jsonl")
step "staging Lean into ${SHM}"
kimina_stage "${SHM}"
key=$(head -c 24 /dev/urandom | base64 | tr -d '/+=')
url="http://127.0.0.1:8100"
confine=no
if kimina_confinable; then confine=yes; fi
export PYTHONPATH="${KIMINA_INSTALL}/kimina-lean-server/client"
for attempt in 1 2; do
    step "Kimina server, ${KIMINA_REPLS:-128} REPLs, bubblewrap: ${confine}"
    kimina_serve 127.0.0.1 8100 "${KIMINA_REPLS:-128}" "${header}" "${key}" "${LEAN_DATA}/logs/${SLURM_JOB_ID}-kimina-server-${confine}.log" "${confine}"
    if kimina_wait "${url}" 900 && "${PY}" "${LEAN_KIMINA_DIR}/check_server.py" controls "${url}" "${key}" "${header}"; then
        break
    fi
    kill "${KIMINA_PID}" 2>/dev/null || true
    sleep 3
    pkill -u "${USER}" -f "$(kimina_repl_bin)" 2>/dev/null || true
    [[ "${confine}" == yes ]] || { echo "ERROR: the Kimina server failed its controls" >&2; exit 1; }
    echo "WARNING: the controls failed with REPLs in bubblewrap; trying unconfined (see the server log)" >&2
    confine=no
done
for split in train validation; do
    step "validating ${split} statements on Lean ${REPL_TAG}"
    "${PY}" "${LEAN_KIMINA_DIR}/validate_statements.py" --url "${url}" --api-key "${key}" \
        --max-inflight "${KIMINA_REPLS:-128}" "${DATA}/${split}.jsonl" "${DATA}/${split}.v434.jsonl.part"
    mv "${DATA}/${split}.v434.jsonl.part.dropped.jsonl" "${DATA}/${split}.v434.jsonl.dropped.jsonl"
    mv "${DATA}/${split}.v434.jsonl.part" "${DATA}/${split}.v434.jsonl"
done
step "verified Nemotron proofs (Lean 4.12) built as Gym does, checked on Lean ${REPL_TAG}"
"${PY}" "${LEAN_KIMINA_DIR}/check_server.py" samples "${url}" "${key}" "${GYM}/app.py" "${DATA}/nemotron_sample_proofs.jsonl" 400 || true
curl -s "${url}/stats"; echo
wc -l "${DATA}"/*.jsonl
step "done"
