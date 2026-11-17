#!/usr/bin/env python3
"""Gym rows of the Lean task from Goedel-Prover's RL dataset (Goedel-LM/RL_dataset_V2), for training with the
Kimina verifier (train.sh). Needs pyarrow; downloads the dataset's three parquet files (336 MB) into --cache once.
    python3 goedel_rows.py --cache /tmp/goedel --n 3000 --seed 1 --exclude-sample 500:0 \
        --exclude-statements $SCRATCH/lean/data/validation.v434.jsonl goedel-train-s1.jsonl
Then keep the rows whose statement elaborates on our Lean (validate_statements.py, see RUNBOOK.md).

1. Sample: the lean4_completion subset at the pinned revision, one user message each ("Complete the following
   Lean 4 code:" + a theorem ending in `:= by sorry`), duplicate statements removed (49,913 remain), then --n of
   them with random.Random(--seed).sample, in dataset order. Each source with fewer than 10 problems in the
   sample is reported as "other". This is chonk-tests' goedel_sample.py, so the same --n and --seed give its rows.
2. Convert: the prompt is the dataset's own user message, unchanged; header + formal_statement is the code before
   the proof, with `:= by sorry` replaced by `:= by` and a newline (as in train.v434.jsonl). agent_ref and
   informal_prefix are those of the Lean task's other rows (Gym routes a row to the agent its agent_ref names).
3. Filter:
   - statements with a `:=` before their final `:= by` (a `let x := ...`): Gym's extract_proof_only cuts the
     model's code at the first `:=` after `theorem`, so it would score such a statement 0 whatever the model wrote;
   - --exclude-sample N:SEED: the statements of another sample drawn the same way (500:0 is chonk-tests' evaluation
     sample), so that evaluation stays clean;
   - --exclude-statements (repeatable): any statement of these files (e.g. the validation set), compared without names,
     comments and whitespace;
   - duplicates by the same comparison.
"""
import argparse
import collections
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

REPO = "Goedel-LM/RL_dataset_V2"
REVISION = "5c99dce377f387bd271d726f205e69380b792a9d"  # main on 2026-11-16
FILES = [f"data/train-0000{i}-of-00003.parquet" for i in range(3)]
FENCE = "`" * 3
CODE = re.compile(FENCE + r"lean4?[ \t]*\n(.*?)" + FENCE, re.S)
TARGET = re.compile(r"^(?:theorem|lemma)\s+(\S+)", re.M)
SORRY_TAIL = re.compile(r":=\s*(?:by\s*)?sorry\s*$")
AGENT_REF = {"type": "responses_api_agents", "name": "math_formal_lean_simple_agent"}


def download(cache):
    paths = []
    for name in FILES:
        path = cache / REVISION[:12] / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
            print(f"downloading {url}", file=sys.stderr)
            partial = path.with_suffix(".part")
            urllib.request.urlretrieve(url, partial)
            partial.rename(path)
        paths.append(path)
    return paths


def split(user):
    """(header, informal, statement, name) from a lean4_completion user message, or None."""
    blocks = CODE.findall(user)
    if len(blocks) != 1:
        return None
    code = blocks[0].strip("\n")
    targets = list(TARGET.finditer(code))
    if not targets:
        return None
    start, name = targets[-1].start(), targets[-1][1]
    before = code[:start].rstrip()
    informal = ""
    if before.endswith("-/"):  # the doc comment (or plain comment) that states the problem in words
        opening = before.rfind("/-")
        informal = re.sub(r"^/-+|-+/$", "", before[opening:]).strip()
        start, before = opening, before[:opening].rstrip()
    statement = code[start:].rstrip()
    return before, informal, statement, name


def family(name):
    """The source a theorem name comes from: OMR_problem_348663 -> OMR_problem."""
    return re.sub(r"_?\d+$", "", name) or name


def candidates(cache):
    import pyarrow.parquet as pq

    out, seen, skipped, position = [], set(), 0, -1
    for path in download(cache):
        for row in pq.read_table(path, columns=["subset", "messages"]).to_pylist():
            position += 1
            if row["subset"] != "lean4_completion":
                continue
            messages = row["messages"]
            parts = split(messages[0]["content"]) if len(messages) == 1 and messages[0]["role"] == "user" else None
            if parts is None or parts[2].count("sorry") != 1 or not re.search(r":=\s*by\s*sorry$", parts[2]) \
                    or "sorry" in parts[0]:
                skipped += 1
                continue
            if parts[2] in seen:
                continue
            seen.add(parts[2])
            out.append((position, messages[0]["content"], parts))
    print(f"{len(out)} distinct lean4_completion statements, {skipped} rows skipped as not in the expected form",
          file=sys.stderr)
    return out


def sample(cands, n, seed):
    chosen = sorted(random.Random(seed).sample(cands, n))
    sizes = collections.Counter(family(parts[3]) for _, _, parts in chosen)
    return [(index, user, parts, family(parts[3]) if sizes[family(parts[3])] >= 10 else "other")
            for index, (_, user, parts) in enumerate(chosen, 1)]


def splits_statement(statement):
    """True if Gym's extract_proof_only would take part of this statement for the proof."""
    lines = statement.strip().splitlines()
    start = next((i for i, line in enumerate(lines) if re.match(r"^\s*(theorem|example)\b", line)), None)
    if start is None:
        return False
    first = next((i for i in range(start, len(lines)) if ":=" in lines[i]), None)
    return first is not None and first != len(lines) - 1


def statement_key(statement):
    """The statement without its name, comments and whitespace: equal keys state the same theorem."""
    text = re.sub(r"/-.*?-/", " ", statement, flags=re.S)
    text = re.sub(r"--[^\n]*", " ", text)
    text = SORRY_TAIL.sub(":=", text.strip())
    text = re.sub(r"\s*:=\s*by\s*$", ":=", text)
    text = re.sub(r"^\s*(?:theorem|lemma|example)\s+\S+", "theorem", text.strip())
    return re.sub(r"\s+", "", text)


def gym_row(index, user, parts, source):
    header, _, statement, name = parts
    return {
        "responses_create_params": {"input": [{"role": "user", "content": user}]},
        "header": header + "\n\n",
        "formal_statement": SORRY_TAIL.sub(":= by", statement) + "\n",
        "informal_prefix": "",
        "agent_ref": dict(AGENT_REF),
        "name": name,
        "source": source,
        "goedel_index": index,
        "gym_splits_statement": splits_statement(statement),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache", type=Path, required=True, help="folder for the parquet files")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--exclude-sample", action="append", default=[], metavar="N:SEED")
    p.add_argument("--exclude-statements", action="append", default=[], metavar="JSONL")
    p.add_argument("dst")
    args = p.parse_args(argv)
    cands = candidates(args.cache)
    rows = [gym_row(*s) for s in sample(cands, args.n, args.seed)]
    names = set()
    for spec in args.exclude_sample:
        n, seed = (int(x) for x in spec.split(":"))
        names |= {parts[3] for _, _, parts, _ in sample(cands, n, seed)}
    keys = set()
    for path in args.exclude_statements:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    keys.add(statement_key(r.get("formal_statement") or r["formal_stmt"]))
    dropped = {"gym_splits_statement": 0, "excluded sample": 0, "excluded statement": 0, "duplicate statement": 0}
    kept, seen = [], set()
    for row in rows:
        key = statement_key(row["formal_statement"])
        if row["gym_splits_statement"]:
            dropped["gym_splits_statement"] += 1
        elif row["name"] in names:
            dropped["excluded sample"] += 1
        elif key in keys:
            dropped["excluded statement"] += 1
        elif key in seen:
            dropped["duplicate statement"] += 1
        else:
            seen.add(key)
            kept.append(row)
    with open(args.dst, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(kept)} of {len(rows)} sampled rows -> {args.dst}; dropped {dropped}", file=sys.stderr)


if __name__ == "__main__":
    main()
