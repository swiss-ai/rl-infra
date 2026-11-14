#!/bin/bash
# Build the Harbor verified-task set for envs/harbor-verified-sft.yaml, on a login node (no GPU,
# container or Slurm; standard library only):
#   bash envs/prepare_harbor_verified_tasks.sh <store> <harbor_data_dir>
# <store> is the harbor_verified_tasks store, one directory per validation campaign with its tasks
# as tarballs in <campaign>/tasks/ (or a single campaign directory). It writes
# <harbor_data_dir>/tasks/<campaign>__<task>/ plus train.jsonl, validation.jsonl and task_names.txt;
# every row's instance_id is `verified::<campaign>__<task>`. Port of
# swiss-ai/nemo-rl-development's prepare_dataset_harbor_agent_osb.sh.
#
# The extraction is incremental: a task whose tarball is unchanged is left alone, and tasks that left
# the store are removed. Every task.toml already names digest-pinned agent and verifier images in a
# private registry, so nothing is pulled here (the run passes the registry credentials).
#
# The collector takes only the tasks a campaign's tasks.csv marks `keep`, and skips a campaign that
# has no tasks.csv. The copy on HANGANG (~/projects/harbor_verified_tasks) has none, so there it needs
# COLLECT_ARGS=--include-unverified. Other knobs: COLLECT_ARGS (also --include-dropped, --force),
# LIMIT (0 = every task), NUM_VALIDATION (16), SEED (0).
#
# The tasks set [verifier] environment_mode = "separate", which needs Harbor 0.25.0 (Gym patch 0011).
set -euo pipefail

[[ $# -eq 2 ]] || { sed -n '2,4s/^# \{0,1\}//p' "$0"; exit 1; }
STORE="$1"
HARBOR_ROOT="$2"
ENVS="$(cd "$(dirname "$0")" && pwd)"
# python3.12 where it exists (HANGANG); Alps login nodes have python3.14 only. PYTHON overrides.
PY="${PYTHON:-$(command -v python3.12 || command -v python3.13 || command -v python3.14 || true)}"
[[ -n "${PY}" ]] || { echo "ERROR: no python3.12, 3.13 or 3.14 on PATH; set PYTHON" >&2; exit 1; }

# shellcheck disable=SC2086  # COLLECT_ARGS is a list of flags.
"${PY}" "${ENVS}/collect_harbor_verified_tasks.py" \
    --verified-dir "${STORE}" \
    --out-dir "${HARBOR_ROOT}/tasks" \
    --limit "${LIMIT:-0}" \
    ${COLLECT_ARGS:-}

# --limit 0 is "no limit" to the collector but a hard cap to the split, so the split gets the count.
NUM_TASKS="$(find "${HARBOR_ROOT}/tasks" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | wc -l)"

"${PY}" "${ENVS}/prepare_harbor_osb_dataset.py" \
    --tasks-dir "${HARBOR_ROOT}/tasks" \
    --output-dir "${HARBOR_ROOT}" \
    --alias verified \
    --agent-name harbor_agent \
    --limit "${NUM_TASKS}" \
    --num-validation "${NUM_VALIDATION:-16}" \
    --seed "${SEED:-0}" \
    --temperature 1.0 \
    --top-p 1.0
