#!/bin/bash
# Download the judge model to scratch; envs/ultrafeedback-rubric.yaml reads it from there. Safe to
# re-run: finished files are skipped. Plain curl, because a login node has no huggingface_hub, and
# 4 files at a time, because one connection stays near 50 MB/s from Hugging Face on HANGANG.
#   bash envs/stage_judge_model.sh [repo] [dest]
set -euo pipefail

REPO="${1:-Qwen/Qwen3.8-27B-FP8}"
DEST="${2:-${SCRATCH:-/scratch/${USER}}/models/$(basename "${REPO}")}"
mkdir -p "${DEST}"

curl -fsS "https://huggingface.co/api/models/${REPO}?blobs=true" | python3 -c '
import json, sys
for s in json.load(sys.stdin)["siblings"]:
    print(s["rfilename"], s.get("size") or 0)
' | REPO="${REPO}" DEST="${DEST}" xargs -n 2 -P "${JOBS:-4}" bash -c '
    file=$1 size=$2
    if [[ -s "${DEST}/${file}" && "$(stat -c %s "${DEST}/${file}")" == "${size}" ]]; then exit 0; fi
    echo "== ${file}"
    curl -fsSL --retry 5 --create-dirs -C - -o "${DEST}/${file}" "https://huggingface.co/${REPO}/resolve/main/${file}"
' _
echo "== ${REPO} in ${DEST}"
