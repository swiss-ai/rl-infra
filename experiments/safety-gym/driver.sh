#!/usr/bin/env bash
# Runs inside the allocated job, then hands execution to the unmodified stock driver.
set -euo pipefail
if [[ "${SAFETY_DATA_PREBUILT:-0}" == 1 ]]; then
  for f in train.jsonl validation.jsonl manifest.json; do [[ -s "${SAFETY_DATA_DIR}/$f" ]] || { echo "Missing prebuilt $f in ${SAFETY_DATA_DIR}" >&2; exit 1; }; done
else
  /usr/bin/python3.12 "${SAFETY_EXPERIMENT_DIR}/prepare_data.py" \
    --source "${SAFETY_DATA_SOURCE}" --output "${SAFETY_DATA_DIR}" \
    --validation-samples "${SAFETY_VALIDATION_SAMPLES}"
fi
SAFETY_JUDGE_WAIT_SECONDS=900 /usr/bin/python3.12 "${SAFETY_EXPERIMENT_DIR}/probe_judge.py"
exec bash "${SAFETY_EXPERIMENT_DIR}/../../backends/nemo-rl/apertus2/driver.sh" "$@"
