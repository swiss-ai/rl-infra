#!/usr/bin/env python3
"""Score one validation of eval/run.sh: pass@1 per corrected-miniF2F split, and optionally the paired difference to
another validation of the same rows (e.g. `start`).
    python3 score.py <val_data_step0.jsonl> <minif2f-corrected-x2.v434.jsonl> [<other val_data_step0.jsonl> ...]
Row i of a validation log (its `idx`) is row i of the validation file. pass@1 = mean reward over a split's rows.
With other logs: per statement, mean over this log's samples minus mean over the others' samples; SE over statements.
"""
import collections
import json
import sys


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        x = r["rewards"]
        out[r["idx"]] = float(x[0] if isinstance(x, list) else x)
    return out


val = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8")]
main = load(sys.argv[1])
assert sorted(main) == list(range(len(val))), f"{sys.argv[1]}: {len(main)} rows, expected {len(val)}"
others = [load(p) for p in sys.argv[3:]]


def per_statement(logs, split):
    d = collections.defaultdict(list)
    for log in logs:
        for i, v in enumerate(val):
            if v["split"] == split and i in log:
                d[v["name"]].append(log[i])
    return {k: sum(x) / len(x) for k, x in d.items()}


print(f"{'split':6s} {'statements':>10s} {'samples':>8s} {'pass@1':>8s}" + ("   diff vs others ± SE" if others else ""))
for split in ("test", "valid"):
    rows = [i for i, v in enumerate(val) if v["split"] == split]
    acc = sum(main[i] for i in rows) / len(rows)
    a = per_statement([main], split)
    line = f"{split:6s} {len(a):10d} {len(rows) // len(a):8d} {100 * acc:7.2f}%"
    if others:
        b = per_statement(others, split)
        d = [a[k] - b[k] for k in a if k in b]
        m = sum(d) / len(d)
        se = (sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5
        line += f"   {100 * m:+.2f} ± {100 * se:.2f}"
    print(line)
