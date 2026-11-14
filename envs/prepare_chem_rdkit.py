#!/usr/bin/env python3
"""Train and validation rows for envs/chem-rdkit-sft.yaml, from the RDKit chemistry generator's native output.

The generator (Litmus-Bench v0.1 clone; molecules from COCONUT, CC0, minus its ChEMBL collection, and the NCI DTP
Open set, US public domain; answers computed with RDKit 2025.09.5; questions are Qwen3-235B-A22B-Instruct-2507
rewrites of hand-checked seeds) writes native_train.jsonl and native_heldout.jsonl, one {"input", "output"} row
each, plus native_meta_*.jsonl with the property, SMILES and provenance of every row. Held-out molecules share no
Bemis-Murcko scaffold with train ones. Answers are integers (counts) or 0/1 (rule checks), so NeMo-RL's math
environment grades the claim frame as is.

This script checks the rows and writes <out>/train.jsonl (all train rows, or --train-rows of them) and
<out>/validation.jsonl (--val-rows held-out rows, one property at a time in turn so every property is in it).
Stdlib only, for the login node:
    python3.12 envs/prepare_chem_rdkit.py <generator output dir> $SCRATCH/chem-rdkit/native
"""
import argparse
import collections
import json
import os
import random
import re
import sys

INT = re.compile(r"-?\d+")


def load(d, split):
    rows = [json.loads(line) for line in open(os.path.join(d, f"native_{split}.jsonl"))]
    meta = [json.loads(line) for line in open(os.path.join(d, f"native_meta_{split}.jsonl"))]
    if len(rows) != len(meta):
        sys.exit(f"{split}: {len(rows)} rows but {len(meta)} metadata lines")
    for i, (r, m) in enumerate(zip(rows, meta)):
        if set(r) != {"input", "output"} or not INT.fullmatch(r["output"]):
            sys.exit(f"{split} row {i}: expected {{input, output}} with an integer output, got {r}")
        if m["smiles"] not in r["input"]:
            sys.exit(f"{split} row {i}: SMILES missing from the prompt")
    return rows, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", help="generator output dir with native_{train,heldout}.jsonl and native_meta_*.jsonl")
    ap.add_argument("out")
    ap.add_argument("--train-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--val-rows", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    train, _ = load(args.src, "train")
    held, held_meta = load(args.src, "heldout")
    if args.train_rows:
        if args.train_rows > len(train):
            sys.exit(f"asked for {args.train_rows} train rows, only {len(train)} exist")
        train = rng.sample(train, args.train_rows)

    by_prop = collections.defaultdict(list)
    for r, m in zip(held, held_meta):
        by_prop[m["property"]].append(r)
    for v in by_prop.values():
        rng.shuffle(v)
    if args.val_rows > len(held):
        sys.exit(f"asked for {args.val_rows} validation rows, only {len(held)} held-out rows exist")
    val, val_props, props = [], set(), sorted(by_prop)
    while len(val) < args.val_rows:
        for p in props:
            if by_prop[p] and len(val) < args.val_rows:
                val.append(by_prop[p].pop())
                val_props.add(p)

    os.makedirs(args.out, exist_ok=True)
    for name, rows in (("train", train), ("validation", val)):
        with open(os.path.join(args.out, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    longest = max(len(r["input"]) for r in train + val)
    zero = sum(r["output"] == "0" for r in train) / len(train)
    print(f"train {len(train)}, validation {len(val)} covering {len(val_props)} of {len(props)} properties; "
          f"longest prompt {longest} characters; answer 0 in {zero:.0%} of train rows -> {args.out}")


if __name__ == "__main__":
    main()
