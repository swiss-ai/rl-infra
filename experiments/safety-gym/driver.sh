#!/usr/bin/env bash
# Runs inside the allocated job, then hands execution to the unmodified stock driver.
set -euo pipefail
/usr/bin/python3.12 "${SAFETY_EXPERIMENT_DIR}/prepare_data.py" \
  --source "${SAFETY_DATA_SOURCE}" --output "${SAFETY_DATA_DIR}" \
  --validation-samples "${SAFETY_VALIDATION_SAMPLES}"
SAFETY_JUDGE_WAIT_SECONDS=900 /usr/bin/python3.12 "${SAFETY_EXPERIMENT_DIR}/probe_judge.py"
exec bash "${SAFETY_EXPERIMENT_DIR}/../../backends/nemo-rl/apertus2/driver.sh" "$@"
