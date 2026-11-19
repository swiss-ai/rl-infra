#!/bin/bash
#SBATCH --job-name=export-hf
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
# HF export of a checkpoint of this task, as for the Lean expert of experts round 1 (8 min on one GH200 node):
# NeMo-RL examples/converters/convert_megatron_to_hf.py, run with rl-infra's driver in the NeMo-RL container.
#   sbatch --output=$SCRATCH/lean/logs/%j-export.out \
#       --export=ALL,RL_INFRA=$PWD,RUN=<run>,STEP=<step>,OUT=<new folder> backends/nemo-rl/lean-kimina/export_hf.sbatch
# The single-process export fails with DeepEP ("DeepEP backend is only supported with flex token dispatcher"), so it
# reads a shadow copy of the checkpoint folder: links to its shards and a run_config.yaml with moe_enable_deepep false
# and moe_flex_dispatcher_backend null (the weights are the same). Afterwards, as for the expert:
#   - replace config.json, configuration_apertus2.py, generation_config.json, tokenizer files and
#     special_tokens_map.json with those of the start model (keep the converter's in another folder; it also adds
#     modeling_apertus2.py and __init__.py);
#   - python3 backends/nemo-rl/lean-kimina/compare_safetensors.py <OUT> <start model>  (names, shapes, dtypes)
set -euo pipefail
: "${RL_INFRA:?RL_INFRA=<rl-infra checkout>}" "${RUN:?}" "${STEP:?}" "${OUT:?OUT=<new folder for the HF checkpoint>}"
CKPT=${SCRATCH}/nemo-rl/results/${RUN}/step_${STEP}
SRC=${CKPT}/policy/weights/iter_0000000
SHADOW=${SCRATCH}/lean/export-${RUN}-step${STEP}/iter_0000000
[[ -d "${SRC}" ]] || { echo "ERROR: no ${SRC}" >&2; exit 1; }
[[ -e "${OUT}" ]] && { echo "ERROR: ${OUT} exists" >&2; exit 1; }

mkdir -p "${SHADOW}"
for f in "${SRC}"/* "${SRC}"/.[!.]*; do
    [[ -e "${f}" ]] || continue
    [[ "$(basename "${f}")" == run_config.yaml ]] && continue
    ln -sfn "${f}" "${SHADOW}/$(basename "${f}")"
done
sed -E -e 's/^( *moe_enable_deepep:).*/\1 false/' -e 's/^( *moe_flex_dispatcher_backend:).*/\1 null/' \
    "${SRC}/run_config.yaml" > "${SHADOW}/run_config.yaml"
grep -E "moe_enable_deepep|moe_flex_dispatcher_backend" "${SHADOW}/run_config.yaml"
echo "shadow: $(ls -A "${SHADOW}" | wc -l) entries, source $(ls -A "${SRC}" | wc -l)"

date
srun --environment=nemo-rl-apertus2 bash -c "cd /opt/nemo-rl && bash ${RL_INFRA}/backends/nemo-rl/apertus2/driver.sh \
    examples/converters/convert_megatron_to_hf.py --config ${CKPT}/config.yaml \
    --megatron-ckpt-path ${SHADOW} --hf-ckpt-path ${OUT}"
date
ls -la "${OUT}" | head -20
du -sh "${OUT}"
