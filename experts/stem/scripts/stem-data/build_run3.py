"""Writes run3/numina_clean.jsonl: the cleaned NuminaMath-1.5 pool (pool D of math_stats.py, all answer types),
wrapped in the DAPO prompt, shuffled with seed 42. Reuses math_stats.py up to the cleaned pool, unchanged."""
import json, random
SRC = "/scratch/hanint06/data/stem-rl/scripts/stats/math_stats.py"
src = open(SRC).read()
exec(compile(src[:src.index("# what each fix changed")], SRC, "exec"))
rows = [{"input": DAPO_HEAD + p.strip() + DAPO_TAIL, "output": a.strip()} for p, a in zip(D["problem"], D["answer"].astype(str))]
random.Random(42).shuffle(rows)
out = BASE + "/run3/numina_clean.jsonl"
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote", out, len(rows))
