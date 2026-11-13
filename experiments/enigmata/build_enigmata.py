"""Turn the Enigmata pool (gen_enigmata.py) into NeMo-RL ResponseDataset files.

Each row: {"input": <the generator's prompt, unchanged>, "output": <JSON {"task", "difficulty",
"answer", "meta"}>, "task", "difficulty"}. The native enigmata environment reads "output" as the
ground truth and passes answer and meta to the task's verifier.

  sample: per (task, difficulty), --per-cell puzzles for the offline pass-rate test
    python3 -I build_enigmata.py sample --pool <dir> --out <file> --per-cell 8
  train: the cells of a selection file, excluding the pass-rate sample puzzles
    python3 -I build_enigmata.py train --pool <dir> --exclude <sample file> --select <json> --out <file>
  The selection file is {"cells": {"<task>/<difficulty>": <rows>, ...}}; a cell with fewer rows
  than asked gives all it has, and the shortfall is printed. --max-prompt-tokens drops longer
  prompts (none by default) and prints how many.
"""

import argparse
import json
import os
import random


def load_pool(pool_dir):
    cells = {}
    for fn in sorted(os.listdir(pool_dir)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(pool_dir, fn)) as f:
            for line in f:
                r = json.loads(line)
                cells.setdefault(f"{r['task']}/{r['difficulty']}", []).append(r)
    return cells


def to_row(r):
    spec = {"task": r["task"], "difficulty": r["difficulty"], "answer": r["answer"], "meta": r["meta"]}
    return {"input": r["prompt"], "output": json.dumps(spec, ensure_ascii=False),
            "task": r["task"], "difficulty": r["difficulty"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sample", "train"])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--select", default=None)
    ap.add_argument("--max-prompt-tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cells = load_pool(args.pool)
    rng = random.Random(args.seed)
    for key in sorted(cells):
        rng.shuffle(cells[key])

    out_rows = []
    if args.mode == "sample":
        for key in sorted(cells):
            out_rows += [to_row(r) for r in cells[key][: args.per_cell]]
        print(f"{len(out_rows)} rows from {len(cells)} cells")
    else:
        excluded = set()
        if args.exclude:
            with open(args.exclude) as f:
                excluded = {json.loads(l)["input"] for l in f}
        with open(args.select) as f:
            select = json.load(f)["cells"]
        dropped_long = 0
        for key, want in sorted(select.items()):
            pool = [r for r in cells.get(key, []) if r["prompt"] not in excluded]
            if args.max_prompt_tokens is not None:
                keep = [r for r in pool if (r.get("prompt_tokens") or 0) <= args.max_prompt_tokens]
                dropped_long += len(pool) - len(keep)
                pool = keep
            take = pool[:want]
            if len(take) < want:
                print(f"{key}: {len(take)} of {want} rows available")
            out_rows += [to_row(r) for r in take]
        if args.max_prompt_tokens is not None:
            print(f"dropped {dropped_long} prompts over {args.max_prompt_tokens} tokens")
        rng.shuffle(out_rows)
        print(f"{len(out_rows)} rows from {len(select)} cells")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
