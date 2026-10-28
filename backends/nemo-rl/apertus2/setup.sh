#!/bin/bash
# One-time setup of the Apertus 2 NeMo-RL stack on the login node (safe to re-run):
#   1. clone the NeMo-RL fork at the pin, merge the pinned PRs, apply our patches (Alps ray.sub,
#      refit fixes) and check out the Automodel submodule the DTensor (HF) trainer imports (+ our loader patch)
#      and the Megatron-LM-MoE / Megatron-Bridge sources the Megatron trainer imports
#   2. render the EDF into ~/.edf/<EDF_NAME>.toml (the old one is backed up if it differs)
#   3. images: stage the scratch working copy jobs read from its capstor master; when that master
#      is missing, import the base image from JFrog (enroot, needs your JFrog login) for
#      uccl-fix/build_image.sh
# SKIP_VLLM_IMAGE=1 skips the standalone vLLM image (not needed for NeMo-RL runs).
set -euo pipefail

APERTUS2_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${APERTUS2_DIR}/../../.." && pwd)"
source "${APERTUS2_DIR}/env"
source "${ROOT}/site.env"
source "${PIN_FILE}"
GIT=(git -c user.name="$(git -C "${ROOT}" config user.name || echo rl-infra)" -c user.email="$(git -C "${ROOT}" config user.email || echo rl-infra@localhost)" -c commit.gpgsign=false)

# 1. NeMo-RL source
if [[ ! -d "${NEMO_RL_DIR}/.git" ]]; then
    echo "== cloning NeMo-RL into ${NEMO_RL_DIR}"
    git clone --filter=blob:none "${NEMO_RL_URL}" "${NEMO_RL_DIR}"
    git -C "${NEMO_RL_DIR}" switch -c "${NEMO_RL_BRANCH}" "${NEMO_RL_COMMIT}"
    while read -r repo pr sha; do
        git -C "${NEMO_RL_DIR}" fetch "https://github.com/${repo}.git" "pull/${pr}/head"
        if [[ "${repo}" == NVIDIA-NeMo/RL && "${pr}" == 2517 ]]; then
            "${GIT[@]}" -C "${NEMO_RL_DIR}" merge --no-commit --no-ff -Xtheirs "${sha:-FETCH_HEAD}" || true
            git -C "${NEMO_RL_DIR}" restore --source=HEAD -- tests/unit/test_recipes_and_test_suites.py
            git -C "${NEMO_RL_DIR}" add tests/unit/test_recipes_and_test_suites.py
            "${GIT[@]}" -C "${NEMO_RL_DIR}" commit -m "Merge ${repo} PR #${pr}"
        else
            "${GIT[@]}" -C "${NEMO_RL_DIR}" merge --no-edit -m "Merge ${repo} PR #${pr}" "${sha:-FETCH_HEAD}"
        fi
    done <<< "${NEMO_RL_PRS}"
    "${GIT[@]}" -C "${NEMO_RL_DIR}" am -3 "${ROOT}"/backends/nemo-rl/patches/*.patch "${APERTUS2_DIR}"/patches/nemo-rl/*.patch
    git -C "${NEMO_RL_DIR}" remote set-url --push origin no_push
else
    echo "== NeMo-RL clone exists: ${NEMO_RL_DIR} ($(git -C "${NEMO_RL_DIR}" branch --show-current) @ $(git -C "${NEMO_RL_DIR}" rev-parse --short HEAD))"
    # Pinned PRs added to the pin after the clone was made.
    while read -r repo pr sha; do
        [[ -n "${sha}" ]] || continue
        git -C "${NEMO_RL_DIR}" merge-base --is-ancestor "${sha}" HEAD 2>/dev/null && continue
        echo "== merging ${repo} PR #${pr} (${sha:0:9})"
        git -C "${NEMO_RL_DIR}" fetch "https://github.com/${repo}.git" "pull/${pr}/head"
        "${GIT[@]}" -C "${NEMO_RL_DIR}" merge --no-edit -m "Merge ${repo} PR #${pr}" "${sha}"
    done <<< "${NEMO_RL_PRS}"
fi
# Plain clone, not `submodule update`: the pinned commit is only reachable from a full fetch.
AM="${NEMO_RL_DIR}/3rdparty/Automodel-workspace/Automodel"
if [[ "$(git -C "${AM}" rev-parse HEAD 2>/dev/null)" != "${AUTOMODEL_COMMIT}" ]]; then
    echo "== Automodel checkout at ${AUTOMODEL_COMMIT:0:9}"
    rm -rf "${AM}"
    git clone --filter=blob:none https://github.com/NVIDIA-NeMo/Automodel.git "${AM}"
    git -C "${AM}" checkout --detach "${AUTOMODEL_COMMIT}"
fi
# Our Automodel patches go on the working tree (HEAD stays at the pin, so the check above holds).
for p in "${APERTUS2_DIR}"/patches/automodel/*.patch; do
    if git -C "${AM}" apply --reverse --check "${p}" 2>/dev/null; then continue; fi
    git -C "${AM}" apply "${p}"
    echo "== Automodel: applied $(basename "${p}")"
done
# Megatron-LM-MoE and Megatron-Bridge for the Megatron trainer (PYTHONPATH via the EDF).
checkout_at() {   # <url> <dir> <commit> <ref the commit is reachable from>
    local url="$1" dir="$2" commit="$3" ref="$4"
    [[ "$(git -C "${dir}" rev-parse HEAD 2>/dev/null)" == "${commit}" ]] && return 0
    echo "== $(basename "${dir}") at ${commit:0:9}"
    [[ -d "${dir}/.git" ]] || git clone --filter=blob:none --no-checkout "${url}" "${dir}"
    git -C "${dir}" cat-file -e "${commit}^{commit}" 2>/dev/null || git -C "${dir}" fetch origin "${ref}"
    git -C "${dir}" checkout --detach "${commit}"
}
checkout_at "${MEGATRON_URL}" "${MEGATRON_DIR}" "${MEGATRON_COMMIT}" "${MEGATRON_REF}"
checkout_at "${BRIDGE_URL}" "${BRIDGE_DIR}" "${BRIDGE_COMMIT}" "${BRIDGE_REF}"
for p in "${APERTUS2_DIR}"/patches/megatron/*.patch; do
    if git -C "${MEGATRON_DIR}" apply --reverse --check "${p}" 2>/dev/null; then continue; fi
    git -C "${MEGATRON_DIR}" apply "${p}"
    echo "== Megatron-LM: applied $(basename "${p}")"
done

# 2. EDF
EDF_FILE="${HOME}/.edf/${EDF_NAME}.toml"
mkdir -p "${HOME}/.edf"
rendered="$(mktemp)"
sed -e "s|@@IMAGE_WORK@@|${IMAGE_WORK}|g" \
    -e "s|@@NEMO_RL_DIR@@|${NEMO_RL_DIR}|g" \
    -e "s|@@SCRATCH@@|${SCRATCH}|g" \
    -e "s|@@PYDEPS@@|${PYDEPS}|g" \
    -e "s|@@MEGATRON_DIR@@|${MEGATRON_DIR}|g" \
    -e "s|@@BRIDGE_DIR@@|${BRIDGE_DIR}|g" \
    "${APERTUS2_DIR}/edf.toml.template" > "${rendered}"
if [[ -f "${EDF_FILE}" ]] && cmp -s "${rendered}" "${EDF_FILE}"; then
    echo "== EDF up to date: ${EDF_FILE}"
    rm -f "${rendered}"
else
    if [[ -f "${EDF_FILE}" ]]; then
        cp "${EDF_FILE}" "${EDF_FILE}.bak.$(date +%Y%m%d-%H%M%S)"
        echo "== EDF changed; previous version backed up next to it"
    fi
    mv "${rendered}" "${EDF_FILE}"
    echo "== EDF written: ${EDF_FILE}"
fi

# 3. Images. enroot layer cache on scratch, never in the NFS home. The unpack space must be
#    node-local: enroot's overlay mount over the unpacked layers fails on Lustre (EINVAL).
export ENROOT_CACHE_PATH="${SCRATCH}/tmp/enroot/cache" ENROOT_DATA_PATH="${SCRATCH}/tmp/enroot/data"
ENROOT_TMP_BASE="${ENROOT_TMP_BASE:-/tmp}"
stage_image() {   # <ref> <master> <work>
    local ref="$1" master="$2" work="$3" tmp
    if [[ ! -s "${master}" ]]; then
        echo "== importing ${ref}"
        export ENROOT_TEMP_PATH="${ENROOT_TMP_BASE}/enroot-tmp-${USER}.$$"
        mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_TEMP_PATH}" "$(dirname "${work}")"
        tmp="${work}.tmp.$$"
        enroot import -o "${tmp}" "docker://${ref}" || { rm -f "${tmp}"; rm -rf "${ENROOT_TEMP_PATH}"; echo "ERROR: import failed" >&2; return 1; }
        rm -rf "${ENROOT_TEMP_PATH}"
        mv "${tmp}" "${work}"
        echo "== saving master ${master}"
        cp "${work}" "${master}.tmp.$$" && mv "${master}.tmp.$$" "${master}"
    elif [[ ! -s "${work}" ]]; then
        echo "== staging ${work} from master"
        mkdir -p "$(dirname "${work}")"
        tmp="${work}.tmp.$$"
        cp "${master}" "${tmp}" || { rm -f "${tmp}"; echo "ERROR: image copy failed" >&2; return 1; }
        mv "${tmp}" "${work}"
    else
        echo "== image present: ${work}"
    fi
}
IMAGE_WORK_DIR="$(dirname "${IMAGE_WORK}")"
if [ -d "${IMAGE_WORK_DIR}" ]
then
	:
elif [ ! -e "${IMAGE_WORK_DIR}" ]
then
	mkdir -p "${IMAGE_WORK_DIR}"
else
	echo "== Image work path '${IMAGE_WORK_DIR}' already exists and is not a folder." >&2
	exit 1
fi
lfs setstripe -c 16 -S 4M "${IMAGE_WORK_DIR}" 2>/dev/null || true
if [[ -s "${IMAGE_MASTER}" ]]; then
    stage_image "" "${IMAGE_MASTER}" "${IMAGE_WORK}"
else
    # The image jobs run is built from the JFrog one, not imported.
    stage_image "${IMAGE_REF}" "${BASE_IMAGE_MASTER}" "${BASE_IMAGE_WORK}"
    echo "ERROR: ${IMAGE_MASTER} missing; build it with uccl-fix/build_image.sh on a compute node" >&2
    exit 1
fi
[[ "${SKIP_VLLM_IMAGE:-0}" == 1 ]] || stage_image "${VLLM_IMAGE_REF}" "${VLLM_IMAGE_MASTER}" "${VLLM_IMAGE_WORK}"
