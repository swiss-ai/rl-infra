#!/bin/bash
# Body of the import job (import_image.sh submits it).
set -euo pipefail
REPO=igitman/nemo-skills-sandbox TAG=0.7.1-arm64
IMAGE="${OUT}/images/nemo-skills-sandbox-${TAG}.sqsh" CACHE="${OUT}/enroot-cache"
[[ -s "${IMAGE}" ]] && { echo "== image present: ${IMAGE}"; exit 0; }
mkdir -p "${CACHE}"
TOKEN=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${REPO}:pull" \
    | python3 -c "import sys, json; print(json.load(sys.stdin)[\"token\"])")
export TOKEN
curl -sf -H "Authorization: Bearer ${TOKEN}" -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
    "https://registry-1.docker.io/v2/${REPO}/manifests/${TAG}" \
    | python3 -c "import sys, json; [print(l[\"digest\"].split(\":\")[1], l[\"size\"]) for l in json.load(sys.stdin)[\"layers\"]]" \
    > "${CACHE}/layers.txt"
CH=$((32 * 1024 * 1024))
while read -r d size; do
    [[ -s "${CACHE}/${d}" ]] && continue
    dir="${CACHE}/parts-${d}"; mkdir -p "${dir}"
    seq 0 $(( (size - 1) / CH )) | xargs -P 32 -I{} bash -c "s=\$(( {} * ${CH} )); e=\$(( s + ${CH} - 1 )); (( e > ${size} - 1 )) && e=$(( size - 1 ));
        f=${dir}/\$(printf %06d {}); want=\$(( e - s + 1 ));
        for t in 1 2 3 4 5 6; do [[ -s \$f && \$(stat -c %s \$f) -eq \$want ]] && exit 0;
            curl -sSfL --max-time 900 -H \"Authorization: Bearer \${TOKEN}\" -r \$s-\$e -o \$f https://registry-1.docker.io/v2/${REPO}/blobs/sha256:${d} || true; done; exit 1"
    cat "${dir}"/* > "${CACHE}/${d}.tmp"
    echo "${d}  ${CACHE}/${d}.tmp" | sha256sum -c - > /dev/null && mv "${CACHE}/${d}.tmp" "${CACHE}/${d}" && rm -rf "${dir}"
done < "${CACHE}/layers.txt"
export ENROOT_CACHE_PATH="${CACHE}" ENROOT_DATA_PATH="/tmp/enroot-data-${USER}" ENROOT_TEMP_PATH="/tmp/enroot-tmp-${USER}"
mkdir -p "${ENROOT_DATA_PATH}" "${ENROOT_TEMP_PATH}"
enroot import -o "${IMAGE}.tmp" "docker://${REPO}:${TAG}" && mv "${IMAGE}.tmp" "${IMAGE}"
rm -rf "${ENROOT_TEMP_PATH}" "${ENROOT_DATA_PATH}"
ls -la "${IMAGE}"
