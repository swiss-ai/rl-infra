"""Shrink Nemotron-Math-Proofs-v1 (29.5 GB jsonl) to what Gym prepare_nemotron_math_proofs.py reads, saved
with datasets.save_to_disk as {"lean": ...}, so Gym script runs unchanged. Only the first message of each
trajectory is kept: the script classifies trajectories by messages[0]["content"] and reads nothing else from
them. Also writes a few verified proofs per source to check the sandbox Lean version against the data.
    python3 reduce_nemotron.py <lean.jsonl> <out dataset dir> <out proofs.jsonl>
"""
import json, random, sys
from datasets import Dataset, DatasetDict

src, out_dir, proofs_out = sys.argv[1:4]
cols = {k: [] for k in ("formal_statement", "lean_header", "problem", "source", "used_in", "messages")}
proofs, n = [], 0
with open(src, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        msgs = row.get("messages") or []
        for k in cols:
            cols[k].append([{"content": msgs[0].get("content", "")}] if k == "messages" and msgs else ([] if k == "messages" else row.get(k)))
        if len(msgs) >= 2 and len(proofs) < 400 and random.random() < 0.002:
            proofs.append({"lean_header": row.get("lean_header"), "formal_statement": row.get("formal_statement"),
                           "answer": msgs[-1].get("content", ""), "source": row.get("source")})
        n += 1
print("rows", n)
cols["used_in"] = [u or [] for u in cols["used_in"]]
DatasetDict({"lean": Dataset.from_dict(cols)}).save_to_disk(out_dir)
with open(proofs_out, "w") as f:
    for p in proofs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print("saved", out_dir, "proofs", len(proofs))
