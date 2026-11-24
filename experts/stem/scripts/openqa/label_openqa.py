"""Domain labels for the openqa questions: one gemma-4-26B-A4B-it copy per GPU (offline vLLM), same prompt and
label set as the knowledge-mcqa labelling (domain-labels/label_worker.py), no hand-label check.
Usage: python3.12 label_openqa.py <worker> <workers> <dir> <model>; reads <dir>/pool.jsonl, writes <dir>/labels_<w>.jsonl"""
import json, math, os, sys, time
W, NW, D, MODEL = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
LABELS = ["physics", "chemistry", "biology", "medicine", "engineering", "earth_science", "math",
          "computer_science", "law", "economics_business", "social_science", "humanities", "other"]
PROMPT = """Classify the academic domain of this exam question. Answer with exactly one label.

physics       - incl. astronomy, nuclear, optics, materials physics
chemistry     - incl. biochemistry mechanisms, analytical, spectroscopy
biology       - incl. genetics, ecology, molecular and cell biology
medicine      - clinical, pharmacology, public health, nursing
engineering   - electrical, mechanical, civil, chemical, materials
earth_science - geology, climate, oceanography, remote sensing
math          - incl. statistics, probability
computer_science
law
economics_business
social_science - psychology, sociology, politics, education
humanities    - history, literature, philosophy, linguistics, arts
other

Question:
{question}

Label:"""
rows = [json.loads(l) for l in open(os.path.join(D, "pool.jsonl"))]
n = math.ceil(len(rows) / NW); rows = rows[W * n:(W + 1) * n]
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
t0 = time.time()
llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.88, enforce_eager=True,
          limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0}, max_num_seqs=512, seed=0)
SP = SamplingParams(temperature=0.0, max_tokens=8, structured_outputs=StructuredOutputsParams(choice=LABELS))
convs = [[{"role": "user", "content": PROMPT.format(question=r["text"][:1500])}] for r in rows]
outs = llm.chat(convs, SP, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
with open(os.path.join(D, "labels_%d.jsonl" % W), "w") as f:
    for r, o in zip(rows, outs):
        lab = o.outputs[0].text.strip()
        f.write(json.dumps({"row": r["row"], "label": lab if lab in LABELS else "INVALID"}) + "\n")
print("worker %d: %d rows in %.0f s" % (W, len(rows), time.time() - t0), flush=True)
