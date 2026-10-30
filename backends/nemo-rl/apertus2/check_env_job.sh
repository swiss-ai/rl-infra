#!/bin/bash
# Submit check_env.sh as a one-node batch job in the Apertus 2 container, with the same site
# configuration and container setup launch.sh uses. Output goes to
# $OUT/slurm/<jobid>-check_env.out; the script prints the path and returns right away.
#   bash backends/nemo-rl/apertus2/check_env_job.sh        (DRY_RUN=1 prints the sbatch command)
#   JOB_TIME, PARTITION, ACCOUNT and MODEL (passed on to check_env.sh) override the defaults.
set -euo pipefail

export APERTUS2_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${APERTUS2_DIR}/../../.." && pwd)"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"
source "${ROOT}/backends/nemo-rl/container.sh"

case "${CONTAINER_RUNTIME}" in
    edf)   container_args="--environment=${EDF_NAME}" ;;
    pyxis) container_pyxis_env
           container_args="--container-image=${CONTAINER} --container-mounts=${MOUNTS} ${PYXIS_EXTRA_ARGS} --no-container-mount-home --container-workdir=/opt/nemo-rl" ;;
    *)     echo "ERROR: unknown CONTAINER_RUNTIME '${CONTAINER_RUNTIME}'" >&2; exit 1 ;;
esac
read -r -a extra <<< "${SBATCH_EXTRA}"
flags=(--nodes=1 --account="${ACCOUNT}" --partition="${PARTITION:-${DEFAULT_PARTITION}}" --time="${JOB_TIME:-00:30:00}"
       --job-name=check-env --output="${OUT}/slurm/%j-check_env.out" ${extra[@]+"${extra[@]}"})
wrap="srun ${container_args} bash ${APERTUS2_DIR}/check_env.sh"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'sbatch'; printf ' %q' "${flags[@]}"; printf ' --wrap %q\n' "${wrap}"
    exit 0
fi
mkdir -p "${OUT}/slurm"
cd "${OUT}/slurm"   # submit from scratch (required on HANGANG)
submitted="$(sbatch "${flags[@]}" --wrap "${wrap}")"
echo "${submitted}"
echo "output: ${OUT}/slurm/${submitted##* }-check_env.out"
