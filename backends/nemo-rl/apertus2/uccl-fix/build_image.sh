#!/bin/bash
# Builds the image jobs run (IMAGE_WORK, saved to IMAGE_MASTER): a copy of the JFrog image
# (BASE_IMAGE_WORK) with UCCL rebuilt from the same commit plus apertus2/patches/uccl/*.patch.
# Everything else in the image is left as it is. Run on a compute node outside any container
# (enroot unpacks the image into node-local /tmp); takes about five minutes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APERTUS2_DIR="$(cd "${HERE}/.." && pwd)"
source "${APERTUS2_DIR}/env"

SRC_IMAGE="${BASE_IMAGE_WORK}"
OUT_IMAGE="${IMAGE_WORK}"
NAME="nemo-rl-ucclfix-$$"
LOG_DIR="${SCRATCH}/nemo-rl/logs/uccl-fix-build"
LOG="${LOG_DIR}/build-$(date +%Y%m%d-%H%M%S).log"

export ENROOT_DATA_PATH="/tmp/enroot-data-${USER}" \
  ENROOT_TEMP_PATH="/tmp/enroot-tmp-${USER}" \
  ENROOT_CACHE_PATH="/tmp/enroot-cache-${USER}" \
  ENROOT_RESTRICT_ENV=y
mkdir -p "${ENROOT_DATA_PATH}" "${ENROOT_TEMP_PATH}" "${ENROOT_CACHE_PATH}" "${LOG_DIR}"

[[ -s "${SRC_IMAGE}" ]] || { echo "ERROR: ${SRC_IMAGE} missing (copy it from ${BASE_IMAGE_MASTER})" >&2; exit 1; }
free_gb="$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)"
if (( free_gb < 150 )); then
  echo "ERROR: only ${free_gb} GB free in /tmp, the unpacked image needs about 100" >&2
  exit 1
fi

cleanup() {
  enroot remove -f "${NAME}" >/dev/null 2>&1 || true
  rm -rf "${ENROOT_TEMP_PATH}"
}
trap cleanup EXIT

{
  echo "== $(date) on $(hostname): ${SRC_IMAGE} -> ${OUT_IMAGE}"
  enroot create -n "${NAME}" "${SRC_IMAGE}"
  rootfs="${ENROOT_DATA_PATH}/${NAME}"
  # Not under /tmp: enroot mounts its own /tmp over the image's.
  mkdir -p "${rootfs}/opt/uccl-fix/patches"
  cp "${HERE}/rebuild_uccl.sh" "${rootfs}/opt/uccl-fix/"
  cp "${APERTUS2_DIR}"/patches/uccl/*.patch "${rootfs}/opt/uccl-fix/patches/"

  echo "== $(date) rebuilding UCCL"
  enroot start --root --rw "${NAME}" bash /opt/uccl-fix/rebuild_uccl.sh /opt/uccl-fix/patches
  rm -rf "${rootfs}/opt/uccl-fix"

  echo "== $(date) exporting"
  tmp_image="${OUT_IMAGE%.sqsh}.tmp.sqsh"
  rm -f "${tmp_image}"
  enroot export -o "${tmp_image}" "${NAME}"
  mv "${tmp_image}" "${OUT_IMAGE}"

  echo "== $(date) saving master ${IMAGE_MASTER}"
  cp "${OUT_IMAGE}" "${IMAGE_MASTER}.tmp.$$"
  mv "${IMAGE_MASTER}.tmp.$$" "${IMAGE_MASTER}"
  echo "== $(date) done: ${OUT_IMAGE}"
} 2>&1 | tee "${LOG}"
