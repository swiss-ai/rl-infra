"""Fix Gym prepare_nemotron_math_proofs.py rows for Gym math_formal_lean, which compiles
header + formal_statement + proof: the script keeps the statement ending in ":= by sorry" and the header
without a final newline, so every proof compiles as "...open Xtheorem ... := by sorry<proof>" and fails.
Rows then end like Gym miniF2F rows: header with a newline, statement with ":= by" and a newline.
In place: python3 fix_rows.py <rows.jsonl>"""
import json, re, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p)]
n = 0
for r in rows:
    h, s = r["header"], r["formal_statement"]
    r["header"] = h.rstrip("\n") + "\n" if h else h
    r["formal_statement"] = re.sub(r"\s*sorry\s*$", "\n", s)
    n += (r["header"], r["formal_statement"]) != (h, s)
with open(p + ".tmp", "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
import os; os.replace(p + ".tmp", p)
print(p, len(rows), "rows,", n, "changed")
