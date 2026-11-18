#!/usr/bin/env bash
# Runs inside the allocated job, then hands execution to the unmodified stock driver.
set -euo pipefail
if [[ "${SAFETY_PROFILE:-legacy}" != legacy ]]; then
  judge_fields="$(SAFETY_ADAPTIVE_JUDGE_MODEL="${SAFETY_ADAPTIVE_JUDGE_MODEL}" /usr/bin/python3.12 /run/wait_adaptive_judge.py "${SAFETY_ADAPTIVE_JUDGE_ENDPOINT_FILE}" "${SAFETY_ADAPTIVE_JUDGE_WAIT_SECONDS:-1200}")"
  mapfile -t judge_values <<<"${judge_fields}"
  [[ "${#judge_values[@]}" == 2 ]] || { echo "Invalid SML judge readiness response." >&2; exit 2; }
  export SAFETY_ADAPTIVE_JUDGE_BASE_URL="${judge_values[0]}"
  export SAFETY_JUDGE_BASE_URL="${judge_values[0]}"
  export SAFETY_JUDGE_MODEL="${judge_values[1]}"
fi
if [[ "${SAFETY_PROFILE:-legacy}" == legacy ]]; then
  /usr/bin/python3.12 "${SAFETY_EXPERIMENT_DIR}/prepare_data.py" \
    --source "${SAFETY_DATA_SOURCE}" --output "${SAFETY_DATA_DIR}" \
    --validation-samples "${SAFETY_VALIDATION_SAMPLES}"
else
  for path in "${SAFETY_DATA_SOURCE}/manifest.json" "${SAFETY_DATA_SOURCE}/adaptive_train.jsonl" "${SAFETY_DATA_SOURCE}/adaptive_validation.jsonl"; do
    [[ -s "${path}" ]] || { echo "Missing immutable adaptive input: ${path}" >&2; exit 2; }
  done
fi
SAFETY_JUDGE_WAIT_SECONDS=900 /usr/bin/python3.12 "${SAFETY_EXPERIMENT_DIR}/probe_judge.py"
exec bash "${SAFETY_EXPERIMENT_DIR}/../../backends/nemo-rl/apertus2/driver.sh" "$@"
