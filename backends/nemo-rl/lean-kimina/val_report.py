#!/usr/bin/env python3
"""Validation report for runs of this task: pass@1 per split of the validation file, for every validation each run
logged, compared with a baseline measured on the same rows (paired per statement; ± = standard error over
statements). Read-only.
    python3 val_report.py --val $LEAN_DATA/data/minif2f-corrected-x2.v434.jsonl \
        --baseline <RUN> [--baseline <RUN> ...] <TAG>=<RUN> [<TAG>=<RUN> ...]
A baseline is the step-0 validation (val_data_step0.jsonl) of the named runs, pooled. Logs are read from
--logs (default $SCRATCH/nemo-rl/logs/<RUN>/exp_*/val_data_step<N>.jsonl). Row i of a log (its `idx`) is taken as
row i of the validation file. Rows need `split` and `name`.
"""
import argparse
import collections
import glob
import json
import os
import re


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        x = r["rewards"]
        out[r["idx"]] = float(x[0] if isinstance(x, list) else x)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--val", required=True, help="the validation file the runs used")
    ap.add_argument("--logs", default=os.path.join(os.environ.get("SCRATCH", ""), "nemo-rl", "logs"))
    ap.add_argument("--baseline", action="append", default=[], metavar="RUN")
    ap.add_argument("runs", nargs="+", metavar="TAG=RUN")
    a = ap.parse_args()
    val = [json.loads(line) for line in open(a.val, encoding="utf-8")]
    splits = sorted({v["split"] for v in val})

    def per_statement(rew):
        d = collections.defaultdict(list)
        for i, v in enumerate(val):
            if i in rew:
                d[(v["split"], v["name"])].append(rew[i])
        return {k: sum(x) / len(x) for k, x in d.items()}

    def files(run):
        fs = glob.glob(os.path.join(a.logs, run, "exp_*", "val_data_step*.jsonl"))
        return sorted(fs, key=lambda f: int(re.search(r"step(\d+)", f)[1]))

    base_rows = collections.defaultdict(list)
    for run in a.baseline:
        for f in files(run):
            if f.endswith("val_data_step0.jsonl"):
                for k, x in per_statement(load(f)).items():
                    base_rows[k].append(x)
    base = {k: sum(x) / len(x) for k, x in base_rows.items()}
    for s in splits:
        b = [x for (sp, _), x in base.items() if sp == s]
        if b:
            print(f"baseline {s}: {100 * sum(b) / len(b):.2f}% ({len(b)} statements, {len(a.baseline)} runs)")
    print(f"{'run':8s} {'step':>4s}  " + "  ".join(f"{s:>7s} {'diff±SE':>13s}" for s in splits))
    for spec in a.runs:
        tag, run = spec.split("=", 1)
        for f in files(run):
            step = int(re.search(r"step(\d+)", f)[1])
            ps = per_statement(load(f))
            cols = []
            for s in splits:
                keys = [k for k in ps if k[0] == s]
                acc = sum(ps[k] for k in keys) / len(keys)
                d = [ps[k] - base[k] for k in keys if k in base]
                if len(d) > 1:
                    m = sum(d) / len(d)
                    se = (sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5
                    cols.append(f"{100 * acc:6.2f}% {100 * m:+6.2f}±{100 * se:4.2f}")
                else:
                    cols.append(f"{100 * acc:6.2f}% {'':>13s}")
            print(f"{tag:8s} {step:4d}  " + "  ".join(cols))


if __name__ == "__main__":
    main()
