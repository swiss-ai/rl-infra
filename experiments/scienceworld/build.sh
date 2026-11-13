#!/bin/bash
# Builds the ScienceWorld rows (build_data.py) into ${OUT_DIR}; submit from /scratch/$USER:
#   sbatch --job-name=sciworld-build --export=ALL,CMD="bash <this dir>/build.sh [extra args]" <this dir>/cpu.sbatch
set -euo pipefail
SW=${SW:-/home01/hanint06/sciworld}
HERE=/home01/hanint06/projects/rl-infra-sciworld/experiments/scienceworld
python3 -u ${HERE}/build_data.py --java $SW/jre/bin/java --package $SW/pydeps --out-dir ${OUT_DIR:-$SW/data} \
    --simplifications teleportAction,openDoors,openContainers,selfWateringFlowerPots \
    --train-rows-per-task ${ROWS_PER_TASK:-200} --dev-per-task 5 --sample-per-task 4 --processes 48 "$@"
