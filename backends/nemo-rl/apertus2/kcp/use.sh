# Sourced by run.sh when the task's context selects a KCP row of contexts.tsv: the job runs the Megatron copy with
# the KCP path for the KDA layers and flash-linear-attention 0.5.2. setup.sh (this directory) makes both, once.
# Needs PROJECTS_DIR, OUT, PIN_FILE and CONTAINER_SPEC (sites/load.sh); exports MEGATRON_DIR, PIN_FILE and
# CONTAINER_SPEC, which apertus2/launch.sh keeps (a variable already set wins).
_kcp_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MEGATRON_DIR="${KCP_MEGATRON_DIR:-${PROJECTS_DIR}/megatron-lm-moe-apertus2-kcp}"
_kcp_fla="${KCP_FLA_DIR:-${PROJECTS_DIR}/fla-0.5.2}"
if ! git -C "${MEGATRON_DIR}" rev-parse -q --verify rl-kcp > /dev/null 2>&1 || [[ ! -d "${_kcp_fla}/fla/ops/cp" ]]; then
    if [[ "${PLAN:-0}" == 1 ]]; then
        echo "KCP:     not set up yet (${MEGATRON_DIR} branch rl-kcp, ${_kcp_fla}); a real launch runs kcp/setup.sh first"
        return 0
    fi
    echo "== KCP setup (once): ${MEGATRON_DIR}, ${_kcp_fla}"
    KCP_MEGATRON_DIR="${MEGATRON_DIR}" KCP_FLA_DIR="${_kcp_fla}" bash "${_kcp_here}/setup.sh"
fi
# The stack's setup.sh checks Megatron out at the pin's commit; this copy of the pin names the KCP branch's commit.
_kcp_commit="$(git -C "${MEGATRON_DIR}" rev-parse rl-kcp)"
mkdir -p "${OUT}/kcp"
sed "s/^MEGATRON_COMMIT=.*/MEGATRON_COMMIT=${_kcp_commit}/" "${PIN_FILE}" > "${OUT}/kcp/pin-${_kcp_commit:0:9}"
export PIN_FILE="${OUT}/kcp/pin-${_kcp_commit:0:9}"
# The container spec with flash-linear-attention 0.5.2 (fla.ops.cp) ahead of the image's copy on PYTHONPATH; a
# missing directory would only show inside the job, as the image's older version without fla.ops.cp.
sed "s|:@@MEGATRON_DIR@@:|:@@MEGATRON_DIR@@:${_kcp_fla}:|" "${CONTAINER_SPEC}" > "${OUT}/kcp/edf.toml.template"
grep -q "${_kcp_fla}" "${OUT}/kcp/edf.toml.template" || { echo "ERROR: could not put ${_kcp_fla} on PYTHONPATH" >&2; exit 1; }
export CONTAINER_SPEC="${OUT}/kcp/edf.toml.template"
echo "KCP:     Megatron ${MEGATRON_DIR} at ${_kcp_commit:0:9}, flash-linear-attention ${_kcp_fla}"
