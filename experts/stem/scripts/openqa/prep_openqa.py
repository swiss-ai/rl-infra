"""Step 1 of the openqa debug set (NeMo-RL container: pyarrow + math_verify).
Reads Nemotron-RL-knowledge-openqa train.parquet and writes, in OUT:
  meta.jsonl  one line per row: row, uuid, prompt (the dataset's own user message), question (problem text),
              answer, kind, gold (rule-checkable by math-verify), pass_q235, pass_gptoss
  pool.jsonl  row, uuid, text (question) for the domain-labelling job
"""
import json, re, sys, logging
import pyarrow.parquet as pq
logging.getLogger("math_verify").setLevel(logging.CRITICAL)
from math_verify import parse
from math_verify.parser import LatexExtractionConfig
SRC, OUT = sys.argv[1], sys.argv[2]
t = pq.read_table(SRC).to_pylist()
def kind(a):
    a = a.strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", a): return "number"
    words = re.findall(r"[A-Za-z]{3,}", re.sub(r"\\[a-zA-Z]+", "", a))
    if not words: return "expression"
    if len(words) <= 3: return "short_phrase"
    return "sentence"
HEAD = re.compile(r"^Solve the following problem step by step\. Put your answer inside \\boxed\{\}\.\s*", re.S)
TAIL = re.compile(r"\s*Remember to put your (final )?answer[^\n]*$", re.S)
cfg = (LatexExtractionConfig(),)
n = 0
with open(OUT + "/meta.jsonl", "w") as fm, open(OUT + "/pool.jsonl", "w") as fp:
    for i, x in enumerate(t):
        prompt = x["responses_create_params"]["input"][0]["content"]
        q = TAIL.sub("", HEAD.sub("", prompt)).strip()
        a = str(x["expected_answer"]).strip()
        k = kind(a)
        gold = False
        if k in ("number", "expression"):
            try:
                gold = len(parse("\\boxed{" + a + "}", extraction_config=cfg)) > 0
            except Exception:
                gold = False
        prof = {p["model_hf_path"]: p["pass_rate"] for p in (x["reward_profiles"] or [])}
        rec = {"row": i, "uuid": x["uuid"], "prompt": prompt, "question": q, "answer": a, "kind": k, "gold": gold,
               "pass_q235": prof.get("Qwen/Qwen3-235B-A22B-Instruct-2507"), "pass_gptoss": prof.get("openai/gpt-oss-120b")}
        fm.write(json.dumps(rec) + "\n")
        fp.write(json.dumps({"row": i, "uuid": x["uuid"], "text": q}) + "\n")
        n += 1
print("rows", n)
