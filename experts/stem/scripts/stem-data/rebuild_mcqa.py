"""Resamples mcqa_run2.jsonl from questions Qwen3-30B-A3B did not answer 5/5 (pass rate < 1.0)."""
import glob, json, random, re, sys
import pandas as pd
DL, OUT = sys.argv[1], sys.argv[2]
mc = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DL}/nvidia__Nemotron-RL-knowledge-mcqa/data/train-*.parquet"))], ignore_index=True)
contents = [c["input"][0]["content"] for c in mc["responses_create_params"]]
nopt = [sum(1 for o in opts if any(v is not None for v in o.values())) for opts in mc["options"]]
pr = [p[0]["pass_rate"] if len(p) else None for p in mc["reward_profiles"]]
keep = [i for i, (c, n, a, p) in enumerate(zip(contents, nopt, mc["expected_answer"], pr))
        if n >= 4 and len(c) <= 3500 and isinstance(a, str) and re.fullmatch(r"[A-Z]", a.strip()) and p is not None and p < 1.0]
random.Random(42).shuffle(keep)
sel = keep[:6480]
with open(f"{OUT}/mcqa_run2.jsonl", "w") as f:
    for i in sel:
        f.write(json.dumps({"input": contents[i].strip(), "output": mc["expected_answer"][i].strip()}) + "\n")
print("pool", len(keep), "| options", pd.Series([nopt[i] for i in sel]).value_counts().to_dict(),
      "| pass rate", pd.Series([round(pr[i], 1) for i in sel]).value_counts().sort_index().to_dict(),
      "| answers", pd.Series([mc["expected_answer"][i] for i in sel]).value_counts().head(10).to_dict())
