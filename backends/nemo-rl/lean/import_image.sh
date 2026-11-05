#!/bin/bash
# Import the NeMo-Skills sandbox image (Lean 4.12 + Mathlib, arm64) as a squashfs for start.sh, on one
# intTest node, and print the job ID. A single stream from Docker Hub gets about 1 MB/s here, so every layer
# comes in 32 MB ranges over 32 connections into an enroot layer cache on scratch, and enroot import then
# finds them all cached.
#   bash import_image.sh            (image: $SCRATCH/lean/images/nemo-skills-sandbox-0.7.1-arm64.sqsh)
set -euo pipefail
LEAN_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${LEAN_DIR}/../../.." && pwd)"
source "${ROOT}/sites/load.sh" && rl_infra_load "${ROOT}/backends/nemo-rl/apertus2/env"
OUT="${SCRATCH}/lean"
mkdir -p "${OUT}/logs" "${OUT}/images"
cd "${OUT}"
# shellcheck disable=SC2086
sbatch --parsable --nodes=1 ${SBATCH_EXTRA} --account="${ACCOUNT}" \
    --partition="${PARTITION:-${RUN_PARTITION:-${DEFAULT_PARTITION}}}" --time=01:30:00 \
    --job-name=lean-import --output="${OUT}/logs/%j-import.out" \
    --export=ALL,OUT="${OUT}" --wrap "bash ${LEAN_DIR}/import_image_job.sh"
