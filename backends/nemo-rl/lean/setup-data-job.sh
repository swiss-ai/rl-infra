#!/bin/bash
# Body of the data job (setup-data.sh submits it).
set -euo pipefail
URL=https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1/resolve/97229c590831adfe96202f5cd071d444d535bf91/data/lean.jsonl
SRC="${OUT}/source/lean.jsonl"
if [[ ! -s "${SRC}" ]]; then
    mkdir -p "${OUT}/source/parts"
    python3 "${LEAN_DIR}/download.py" "${URL}" 29525155225 "${OUT}/source/parts" "${SRC}.part"
    mv "${SRC}.part" "${SRC}" && rm -rf "${OUT}/source/parts"
fi
srun --ntasks=1 --cpu-bind=none --container-image="${IMAGE}" --no-container-mount-home \
    --container-mounts="${OUT}:${OUT},${LEAN_DIR}:${LEAN_DIR}:ro,${GYM_LEAN}:${GYM_LEAN}:ro" bash -c "
set -euo pipefail
[[ -d ${OUT}/source/reduced ]] || python3 ${LEAN_DIR}/reduce_nemotron.py ${SRC} ${OUT}/source/reduced ${OUT}/data/nemotron_sample_proofs.jsonl
# A copy of Gym script, which writes into data/ next to itself.
mkdir -p /tmp/gym-lean/data && cp ${GYM_LEAN}/prepare_nemotron_math_proofs.py /tmp/gym-lean/
python3 /tmp/gym-lean/prepare_nemotron_math_proofs.py --dataset-path ${OUT}/source/reduced --output-name nemotron_math_proofs
python3 - <<PY
import json
ref = {\"type\": \"responses_api_agents\", \"name\": \"math_formal_lean_simple_agent\"}
for src, dst in ((\"/tmp/gym-lean/data/nemotron_math_proofs.jsonl\", \"${OUT}/data/train.jsonl\"), (\"${GYM_LEAN}/data/minif2f_test.jsonl\", \"${OUT}/data/validation.jsonl\")):
    with open(src) as f, open(dst, \"w\") as g:
        n = 0
        for line in f:
            row = json.loads(line); row[\"agent_ref\"] = ref; g.write(json.dumps(row, ensure_ascii=False) + \"\n\"); n += 1
    print(dst, n)
PY
python3 ${LEAN_DIR}/fix_rows.py ${OUT}/data/train.jsonl
"
ls -la "${OUT}/data"
