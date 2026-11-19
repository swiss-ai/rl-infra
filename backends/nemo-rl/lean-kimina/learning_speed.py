#!/usr/bin/env python3
"""Read-only analysis of how fast the Lean GRPO runs learn, from their rollout files (RUNBOOK.md, "Learning speed").
    python3 learning_speed.py          (on Clariden; reads $SCRATCH/nemo-rl/results/<RUN>/rollouts/; Python 3.6+)
The runs are the ablations of 2026-11-18 (RUNS below); edit RUNS for others.
A. Revisited prompts (H8, H8S: 736 prompts, a second epoch after resume): per prompt, correct of 16 at the first and
   the second visit.
B. Same prompts, different runs (3965303 LR 2e-6, H1 LR 1e-5, NOSTD LR 2e-6 no std): are the prompts of each
   weight version the same, and the paired reward difference on them, by step range.
Weight version v trained in step v+2 (v >= 1); version 0 in steps 1-2.
"""
import collections
import glob
import json
import os
import re

S = os.environ["SCRATCH"]
R = f"{S}/nemo-rl/results"
RUNS = {"G2e6": "lean-kimina-v02-goedel-9n-20261117", "H1": "lean-kimina-v02-abl-h1-preemptable-20261118",
        "NOSTD": "lean-kimina-v02-abl-nostd-preemptable-20261118-r2", "H8": "lean-kimina-v02-abl-h8-preemptable-20261118",
        "H8S": "lean-kimina-v02-abl-h8s-preemptable-20261118", "H3": "lean-kimina-v02-abl-h3-preemptable-20261118"}


def groups(run):
    """{version: {prompt: [rewards]}}"""
    out = {}
    for f in glob.glob(f"{R}/{run}/rollouts/train_weights*.jsonl"):
        v = int(re.search(r"train_weights(\d+)", f)[1])
        g = collections.defaultdict(list)
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            g[r["responses_create_params"]["input"][0]["content"]].append(float(r["reward"]))
        out[v] = g
    return out


data = {k: groups(v) for k, v in RUNS.items()}
for k, d in data.items():
    n = sum(len(g) for g in d.values())
    print(f"{k:6s} versions {min(d)}..{max(d)}  groups {n}")

print("\n== A. revisited prompts: correct of 16, first visit vs second visit")
for k in ("H8", "H8S", "H3"):
    visits = collections.defaultdict(list)
    for v in sorted(data[k]):
        for p, rw in data[k][v].items():
            if len(rw) == 16:
                visits[p].append((v, sum(x == 1.0 for x in rw)))
    pairs = [(vs[0], vs[1]) for vs in visits.values() if len(vs) >= 2]
    if not pairs:
        print(f"{k}: no prompt seen twice"); continue
    d = [b[1] - a[1] for a, b in pairs]
    m = sum(d) / len(d); se = (sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) ** .5
    first = sum(a[1] for a, _ in pairs) / len(pairs); second = sum(b[1] for _, b in pairs) / len(pairs)
    up = sum(x > 0 for x in d); down = sum(x < 0 for x in d)
    gap = sum(b[0] - a[0] for a, b in pairs) / len(pairs)
    print(f"{k}: {len(pairs)} prompts seen twice (mean {gap:.1f} versions apart): correct/16 {first:.2f} -> {second:.2f} "
          f"(pass rate {100*first/16:.1f}% -> {100*second/16:.1f}%), diff {m:+.2f} ± {se:.2f}; up {up}, down {down}, same {len(d)-up-down}")
    # by first-visit difficulty
    by = collections.defaultdict(list)
    for a, b in pairs:
        c = "0" if a[1] == 0 else ("1-4" if a[1] <= 4 else ("5-11" if a[1] <= 11 else "12-16"))
        by[c].append(b[1] - a[1])
    print("    by first-visit correct: " + ", ".join(f"{c}: n={len(x)} {sum(x)/len(x):+.2f}" for c, x in sorted(by.items())))

print("\n== B. same prompts across runs (by weight version)")
base = data["G2e6"]
for k in ("H1", "NOSTD"):
    same, rows = 0, []
    common = sorted(set(base) & set(data[k]))
    for v in common:
        pa, pb = set(base[v]), set(data[k][v])
        if pa == pb:
            same += 1
            for p in pa:
                if len(base[v][p]) == 16 and len(data[k][v][p]) == 16:
                    rows.append((v, sum(base[v][p]) / 16, sum(data[k][v][p]) / 16))
    print(f"{k} vs G2e6: {same} of {len(common)} common versions have identical prompt sets")
    for lo, hi in ((0, 9), (10, 19), (20, 29), (30, 39), (40, 49)):
        sel = [(a, b) for v, a, b in rows if lo <= v <= hi]
        if len(sel) < 2:
            continue
        d = [b - a for a, b in sel]
        m = sum(d) / len(d); se = (sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) ** .5
        print(f"   versions {lo:2d}-{hi:2d}: {len(sel):4d} prompts  G2e6 {100*sum(a for a,_ in sel)/len(sel):5.1f}%  "
              f"{k} {100*sum(b for _,b in sel)/len(sel):5.1f}%  diff {100*m:+5.1f} ± {100*se:.1f}")
