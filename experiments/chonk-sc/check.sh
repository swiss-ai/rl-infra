#!/bin/bash
# Step 0 for the Chonk Single-Controller runs: static checks on the login node, no GPU, no sbatch.
#   bash experiments/chonk-sc/check.sh [overrides...]      (overrides as you would pass to run-a/-b/-s.sh)
#
#   (b) the merged NeMo-RL clone ($NEMO_RL_DIR): pinned PRs merged, NVIDIA-NeMo/RL#2517 removed the
#       async_engine assert for vLLM EP != TP, f196's refit/membership/token-capture code survived
#       the -Xtheirs merge (grep markers, see common.sh)
#   (c) runs B and S: the Gym checkout carries the current patches/gym set, incl. 0011 + 0012
#   (a) composes the run A, B and S configs exactly as run-a.sh / run-b.sh / run-s.sh will, then resolves and
#       validates them with the clone's own NeMo-RL code (check.py: load_config, the launcher's
#       overrides, MasterConfig, validate_single_controller_config, setup's config-only checks)
#       and prints the resolved values that matter; with yq, also launch.sh's preflight.sh.
# Exits non-zero on any failure.
#
# Env: RUNS ("a b s", default), CHECK_PYTHON (a Python >= 3.12 with omegaconf, hydra-core, pydantic;
# default: a venv in $SCRATCH/chonk-sc-check/venv, created on first use with pip from PyPI),
# HARBOR_ROOT (default $SCRATCH/osb/harbor_micro). Packages the config code imports but the venv
# lacks (torch, ray, vllm, ...) are stubbed by check.py and listed; for a check without stubs run
# it in the job image instead (README.md, "Step 0").
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env" || exit 1
source "${HERE}/common.sh"
RUNS="${RUNS:-a b s}"
WORK="${SCRATCH}/chonk-sc-check"
mkdir -p "${WORK}"
# what the driver resolves ${oc.env:...} against: driver.sh exports RL_INFRA_DIR,
# run-harbor-osb.sh the Harbor paths and the registry placeholders
export SCRATCH RL_INFRA_DIR="${ROOT}"
export HARBOR_ROOT="${HARBOR_ROOT:-${SCRATCH}/osb/harbor_micro}"
export HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:-${HARBOR_ROOT}/tasks}"
export RL_REGISTRY_USERNAME="${RL_REGISTRY_USERNAME:-anonymous}" RL_REGISTRY_PASSWORD="${RL_REGISTRY_PASSWORD:-unused}"
# run S: what run-s.sh and run-harbor-osb.sh export (paths and labels only; the key is never read here)
export OPENSANDBOX_API_KEY_FILE="${OPENSANDBOX_API_KEY_FILE:-${HOME}/.opensandbox_token}"
export NEMO_GYM_USER="${NEMO_GYM_USER:-${USER}}" NEMO_GYM_WORKLOAD="${NEMO_GYM_WORKLOAD:-chonk-harbor-sft-sc}"
export NEMO_GYM_RUN_ID="${NEMO_GYM_RUN_ID:-chonk-sc-check-s}"
total=0

echo "== (b) NeMo-RL clone"
check_nemo_rl_clone; total=$((total + $?))

if [[ " ${RUNS} " == *" b "* || " ${RUNS} " == *" s "* ]]; then
    echo "== (c) Gym checkout for runs B and S (token capture)"
    check_gym_capture; total=$((total + $?))
fi

echo "== (a) composed configs"
pick_python() {
    if [[ -n "${CHECK_PYTHON:-}" ]]; then echo "${CHECK_PYTHON}"; return 0; fi
    local venv="${WORK}/venv" base="" c
    if [[ -x "${venv}/bin/python" ]] && "${venv}/bin/python" -c "import omegaconf, hydra, pydantic" 2>/dev/null; then
        echo "${venv}/bin/python"; return 0
    fi
    # login nodes: python3.14 (the system python3 is older)
    for c in python3.14 python3.13 python3.12 python3; do
        command -v "${c}" >/dev/null || continue
        "${c}" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null && { base="$(command -v "${c}")"; break; }
    done
    [[ -n "${base}" ]] || { echo "ERROR: no Python >= 3.12 on PATH (the image runs NeMo-RL on 3.12); set CHECK_PYTHON, e.g. a uv-managed one, or run step 0 in the image (README.md)" >&2; return 1; }
    echo "   creating ${venv} with ${base} (omegaconf, hydra-core, pydantic at the versions of NeMo-RL's uv.lock)" >&2
    rm -rf "${venv}"
    "${base}" -m venv "${venv}" >&2 && "${venv}/bin/pip" install -q --disable-pip-version-check \
        omegaconf==2.3.1 hydra-core==1.3.2 pydantic==2.13.5 pyyaml >&2 \
        || { echo "ERROR: could not build ${venv}" >&2; return 1; }
    echo "${venv}/bin/python"
}
if ! PY="$(pick_python)"; then
    total=$((total + 1))
else
    echo "   python: ${PY} ($("${PY}" -c 'import sys, pydantic; print(sys.version.split()[0], "pydantic", pydantic.VERSION)'))"
    NODES="$(command -v yq >/dev/null && yq r "${CHONK_RECIPE}" cluster.num_nodes || echo 12)"
    for run in ${RUNS}; do
        cfg="${WORK}/run-${run}.yaml"
        compose_cfg "${run}" "${cfg}"
        name="chonk-sc-check-${run}"
        echo "-- run ${run^^}: ${cfg}"
        sed 's/^/   | /' "${cfg}"
        # the overrides launch.sh appends to the driver command
        ( cd "${WORK}" && "${PY}" -I "${HERE}/check.py" "${NEMO_RL_DIR}" "${cfg}" "${run}" \
            cluster.num_nodes="${NODES}" cluster.gpus_per_node=4 \
            checkpointing.checkpoint_dir="${OUT}/results/${name}" logger.log_dir="${OUT}/logs/${name}" "$@" )
        total=$((total + $?))
        if command -v yq >/dev/null; then
            echo "   preflight.sh (as launch.sh runs it before sbatch):"
            # as launch.sh: the driver's cluster.* overrides first, the run launcher's JOB_TIME default
            case "${run}" in a) jt=01:30:00 ;; b) jt=02:30:00 ;; *) jt=03:00:00 ;; esac
            NEMO_RL_DIR="${NEMO_RL_DIR}" JOB_TIME="${JOB_TIME:-${jt}}" \
                bash "${ROOT}/backends/nemo-rl/preflight.sh" "${cfg}" "${NODES}" \
                cluster.num_nodes="${NODES}" cluster.gpus_per_node=4 "$@" | sed 's/^/   | /'
            (( PIPESTATUS[0] == 0 )) || total=$((total + 1))
        else
            echo "   note  yq not found: preflight.sh skipped (launch.sh runs it at submission)"
        fi
    done
fi

echo "== inputs on scratch (warnings only)"
model="${SCRATCH}/models/apertus2-chonk-lc-iter3576"
[[ -d "${model}" ]] && echo "  ok    model ${model}" || echo "  WARN  model ${model} missing (run.sh, i.e. run A, stages it; run B does not)"
if [[ " ${RUNS} " == *" b "* || " ${RUNS} " == *" s "* ]]; then
    for f in "${HARBOR_ROOT}/train.jsonl" "${HARBOR_ROOT}/validation.jsonl" "${HARBOR_TASKS_DIR}"; do
        [[ -e "${f}" ]] && echo "  ok    ${f}" || echo "  WARN  ${f} missing (runs B and S need it)"
    done
fi
if [[ " ${RUNS} " == *" s "* ]]; then
    [[ -r "${SFT_MODEL_SRC}/config.json" && -r "${SFT_MODEL_SRC}/model.safetensors.index.json" ]] \
        && echo "  ok    SFT export ${SFT_MODEL_SRC}: $(find "${SFT_MODEL_SRC}/" -maxdepth 1 -name '*.safetensors' | wc -l) shards" \
        || echo "  WARN  SFT export ${SFT_MODEL_SRC}: no readable config.json + model.safetensors.index.json (run S needs it)"
    [[ -f "${SFT_MODEL_DIR}/MODELING_SOURCE" ]] && echo "  ok    staged ${SFT_MODEL_DIR}" \
        || echo "  note  ${SFT_MODEL_DIR} not staged yet (run-s.sh stages it)"
    [[ -r "${OPENSANDBOX_API_KEY_FILE}" ]] && echo "  ok    key file ${OPENSANDBOX_API_KEY_FILE}" \
        || echo "  WARN  no OpenSandbox key file ${OPENSANDBOX_API_KEY_FILE} (run S reads it)"
fi

if (( total )); then echo "== step 0: ${total} failure(s)"; exit 1; fi
echo "== step 0: all checks passed"
