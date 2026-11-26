#!/bin/bash
# Build the chisel reference image (see Containerfile, README.md) with podman, test it as an image mount,
# and write an OCI archive that can be pushed with skopeo. Does not push.
#   WORK=<empty or reused dir> bash experiments/agent-image-volumes/build.sh
# Env: WORK (required; build context + archive), CHISEL_VERSION (default 1.12.0),
#      IMAGE (local name, default localhost/agent-bin/chisel:<version>), TEST_IMAGE (any image with a shell,
#      default docker.io/library/busybox:1.36; only for the local mount test, SKIP_TEST=1 skips it).
# Push afterwards (the registry path is yours to choose):
#   skopeo copy oci-archive:$WORK/chisel-<version>.oci.tar docker://<registry>/<project>/chisel:<version>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
CHISEL_VERSION="${CHISEL_VERSION:-1.12.0}"
WORK="${WORK:?set WORK to a build directory (not /tmp on a tmpfs machine)}"
IMAGE="${IMAGE:-localhost/agent-bin/chisel:${CHISEL_VERSION}}"
TEST_IMAGE="${TEST_IMAGE:-docker.io/library/busybox:1.36}"
ctx="${WORK}/context-${CHISEL_VERSION}"
mkdir -p "${ctx}"

# inputs: the binary through the pinned-checksum fetcher the runs use, the licence from the same tag
CHISEL_VERSION="${CHISEL_VERSION}" CHISEL_DIR="${WORK}/chisel" bash "${ROOT}/backends/nemo-rl/apertus2/fetch-chisel.sh"
install -m 0555 "${WORK}/chisel/chisel_${CHISEL_VERSION}_linux_amd64" "${ctx}/chisel"
curl -fsSL --retry 3 -o "${ctx}/LICENSE" "https://raw.githubusercontent.com/jpillora/chisel/v${CHISEL_VERSION}/LICENSE"
grep -q 'MIT License' "${ctx}/LICENSE" || { echo "ERROR: unexpected LICENSE content" >&2; exit 1; }
(cd "${ctx}" && sha256sum chisel LICENSE > SHA256SUMS)
cp "${HERE}/Containerfile" "${ctx}/Containerfile"

podman build --squash --platform linux/amd64 --build-arg "CHISEL_VERSION=${CHISEL_VERSION}" -t "${IMAGE}" "${ctx}"

if [[ "${SKIP_TEST:-0}" != 1 ]]; then
    # the same thing a pod does: the image mounted read-only at a path, used from another container
    podman run --rm --platform linux/amd64 \
        --mount "type=image,source=${IMAGE},destination=/opt/agent-bin/chisel" "${TEST_IMAGE}" sh -euc "
            cd /opt/agent-bin/chisel/${CHISEL_VERSION}
            ls -la
            sha256sum -c SHA256SUMS
            ./chisel --version
            if touch /opt/agent-bin/chisel/probe 2>/dev/null; then echo 'FAIL: mount is writable'; exit 1; fi
            echo 'ok: mount is read-only'"
fi

archive="${WORK}/chisel-${CHISEL_VERSION}.oci.tar"
rm -f "${archive}"
podman save --format oci-archive -o "${archive}" "${IMAGE}"
echo "== ${IMAGE} -> ${archive} ($(du -h "${archive}" | cut -f1))"
echo "   push: skopeo copy oci-archive:${archive} docker://<registry>/<project>/chisel:${CHISEL_VERSION}"
