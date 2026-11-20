"""Generate an Enigmata puzzle pool with the task generators of the Enigmata repository.

One subprocess per (task, difficulty) cell, each seeded from (seed, task, difficulty), so a cell is
reproducible unless its generator draws its own seed (hamiltonian_* do). A cell stops at --count
unique prompts or after --cell-timeout seconds, whichever comes first; what it produced by then is
kept. Every row keeps what the task's verifier needs: the prompt, the answer and the meta exactly as
the generator returned them.

  python3 -I gen_enigmata.py --repo <Enigmata checkout> --out <dir> --count 1000 [--tasks a b ...]

Output: <out>/pool/<task>.jsonl, one row per puzzle:
  {"task", "difficulty", "seed", "prompt", "answer", "meta", "prompt_tokens"}
and <out>/pool/summary.json (rows, seconds and errors per cell). prompt_tokens is the length of the
prompt text under --tokenizer (no system frame), or null without one.

Excluded by default (see handover/enigmata-integration.md): crypto_KKA and crypto_KPA read a
bundled sentence bank of unstated provenance (CoLA-, SNLI/Flickr- and news-like sentences).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

EXCLUDED = {"crypto_KKA", "crypto_KPA"}
DIFFICULTIES = ("easy", "medium", "hard")


def cell_seed(seed: int, task: str, difficulty: str) -> int:
    return int(hashlib.md5(f"{seed}:{task}:{difficulty}".encode()).hexdigest()[:8], 16)


def run_cell(repo: str, task: str, difficulty: str, count: int, seed: int, part: str, pydeps: str | None) -> None:
    """Child process: generate up to `count` unique prompts of one cell into `part`."""
    import importlib
    import random

    import numpy as np

    os.chdir(repo)
    sys.path.insert(0, repo)
    if pydeps:  # python-sat (hamiltonian_*) and python-constraint (hitori)
        sys.path.insert(1, pydeps)
    s = cell_seed(seed, task, difficulty)
    random.seed(s)
    np.random.seed(s)
    module = importlib.import_module(f"verifiable_tasks.tasks.{task}.generator")
    seen = set()
    errors = 0
    # Over-generate: easy cells repeat prompts (e.g. small grids); duplicates are dropped.
    budget = count * 3
    with open(part, "w") as f:
        gen = module.generate(budget, difficulty=difficulty, language="en", split="train")
        while len(seen) < count:
            try:
                row = next(gen)
            except StopIteration:
                break
            except Exception as e:  # a generator that raises mid-stream ends the cell
                errors += 1
                print(f"[{task}/{difficulty}] generator error: {type(e).__name__}: {e}", file=sys.stderr)
                break
            h = hashlib.md5(row["prompt"].encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            f.write(
                json.dumps(
                    {
                        "task": task,
                        "difficulty": difficulty,
                        "seed": s,
                        "prompt": row["prompt"],
                        "answer": row.get("answer"),
                        "meta": row.get("meta"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, required=True, help="unique prompts per task and difficulty")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--cell-timeout", type=float, default=1800.0)
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json for prompt token counts")
    ap.add_argument("--pydeps", default=None, help="directory with python-sat and python-constraint")
    ap.add_argument("--cell", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--part", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.cell:
        task, difficulty = args.cell.split(":")
        run_cell(os.path.abspath(args.repo), task, difficulty, args.count, args.seed, args.part, args.pydeps)
        return

    task_dir = os.path.join(args.repo, "verifiable_tasks", "tasks")
    tasks = args.tasks or sorted(
        t for t in os.listdir(task_dir)
        if os.path.exists(os.path.join(task_dir, t, "generator.py")) and t not in EXCLUDED
    )
    parts_dir = os.path.join(args.out, "parts")
    pool_dir = os.path.join(args.out, "pool")
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(pool_dir, exist_ok=True)

    cells = [(t, d) for t in tasks for d in DIFFICULTIES]
    pending = list(cells)
    running = {}
    info = {}
    t_start = time.time()
    while pending or running:
        while pending and len(running) < args.workers:
            t, d = pending.pop(0)
            part = os.path.join(parts_dir, f"{t}__{d}.jsonl")
            err = open(part + ".err", "w")
            p = subprocess.Popen(
                [sys.executable, "-I", os.path.abspath(__file__), "--repo", args.repo, "--out", args.out,
                 "--count", str(args.count), "--seed", str(args.seed), "--cell", f"{t}:{d}", "--part", part]
                + (["--pydeps", args.pydeps] if args.pydeps else []),
                stdout=subprocess.DEVNULL, stderr=err,
            )
            running[(t, d)] = (p, time.time(), err)
        time.sleep(2)
        for key, (p, t0, err) in list(running.items()):
            rc = p.poll()
            timed_out = rc is None and time.time() - t0 > args.cell_timeout
            if rc is None and not timed_out:
                continue
            if timed_out:
                p.kill()
                p.wait()
            err.close()
            info[key] = {"rc": p.returncode, "timed_out": timed_out, "seconds": round(time.time() - t0, 1)}
            del running[key]
            print(f"{key[0]}/{key[1]}: rc={p.returncode} timed_out={timed_out} "
                  f"{info[key]['seconds']} s ({len(info)}/{len(cells)} cells, {time.time() - t_start:.0f} s)",
                  flush=True)

    tok = None
    if args.tokenizer:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(args.tokenizer)

    summary = {}
    for t in tasks:
        rows_out = 0
        with open(os.path.join(pool_dir, f"{t}.jsonl"), "w") as out:
            seen = set()
            for d in DIFFICULTIES:
                part = os.path.join(parts_dir, f"{t}__{d}.jsonl")
                n = 0
                if os.path.exists(part):
                    with open(part) as f:
                        for line in f:
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError:  # a line cut by a kill
                                continue
                            h = hashlib.md5(row["prompt"].encode()).hexdigest()
                            if h in seen:
                                continue
                            seen.add(h)
                            row["prompt_tokens"] = (
                                len(tok.encode(row["prompt"], add_special_tokens=False).ids) if tok else None
                            )
                            out.write(json.dumps(row, ensure_ascii=False) + "\n")
                            n += 1
                errtail = ""
                if os.path.exists(part + ".err"):
                    with open(part + ".err", errors="replace") as f:
                        lines = [l.rstrip() for l in f if "it/s]" not in l and "s/it]" not in l and l.strip()]
                    errtail = "\n".join(lines[-5:])
                summary[f"{t}/{d}"] = {**info.get((t, d), {}), "rows": n, "stderr_tail": errtail}
                rows_out += n
                for path in (part, part + ".err"):
                    if os.path.exists(path):
                        os.remove(path)
        print(f"{t}: {rows_out} rows", flush=True)
    os.rmdir(parts_dir)
    with open(os.path.join(pool_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: (v["rows"], v.get("seconds"), v.get("timed_out")) for k, v in summary.items()}), flush=True)


if __name__ == "__main__":
    main()
