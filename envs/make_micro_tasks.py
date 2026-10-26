#!/usr/bin/env python3
"""Generate Harbor micro-tasks for the OpenSandbox + Chonk plumbing run.

Each task is a few sentences of instruction, a public image (no registry push, no 1 GB pull),
and a verifier that writes 1 or 0 to /logs/verifier/reward.txt. Every task has a reference
tests-passing solution in solution/solve.sh, used to check the verifier (not used in training).

    python3 make_micro_tasks.py --out <dir>/tasks [--count 96] [--seed 0] [--num-validation 8]

Writes <dir>/tasks/<task>/..., plus <dir>/train.jsonl, <dir>/validation.jsonl and <dir>/task_names.txt
(<dir> is the parent of --out; point HARBOR_ROOT at it). Every row's instance_id is
`swe_lego::<task>`: envs/harbor-osb.yaml maps the dataset alias swe_lego to <dir>/tasks.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import textwrap
from pathlib import Path

IMAGE = "python:3.12-slim"
NAMES = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy",
         "mallory", "niaj", "olivia", "peggy", "rupert", "sybil", "trent", "victor", "walter"]
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
         "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo"]


def fib(n: int) -> list[int]:
    out = [0, 1]
    while len(out) < n:
        out.append(out[-1] + out[-2])
    return out[:n]


def t_sum(r: random.Random):
    nums = [r.randint(2, 99) for _ in range(r.randint(3, 6))]
    ins = f"Write the sum of these numbers to /app/answer.txt (create /app if needed): {', '.join(map(str, nums))}. The file must contain only the number."
    check = f"open('/app/answer.txt').read().strip() == {str(sum(nums))!r}"
    sol = f"mkdir -p /app && echo {sum(nums)} > /app/answer.txt"
    return "sum", ins, check, sol


def t_sorted(r: random.Random):
    names = r.sample(NAMES, r.randint(4, 7))
    ins = f"Create /app/names.txt (create /app if needed) containing these names, one per line, sorted alphabetically: {', '.join(names)}."
    want = "\\n".join(sorted(names))
    check = f"[l.strip() for l in open('/app/names.txt').read().strip().splitlines()] == {sorted(names)!r}"
    sol = f"mkdir -p /app && printf '%s\\n' {' '.join(sorted(names))} > /app/names.txt"
    return "sorted", ins, check, sol


def t_reverse(r: random.Random):
    s = "".join(r.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(r.randint(8, 14)))
    ins = f"Write the reverse of the string {s} to /app/rev.txt (create /app if needed). The file must contain only the reversed string."
    check = f"open('/app/rev.txt').read().strip() == {s[::-1]!r}"
    sol = f"mkdir -p /app && echo {s[::-1]} > /app/rev.txt"
    return "reverse", ins, check, sol


def t_fib(r: random.Random):
    n = r.randint(6, 14)
    ins = (f"Write a Python script /app/fib.py (create /app if needed) that prints the first {n} Fibonacci numbers, "
           "starting from 0, separated by single spaces, on one line.")
    want = " ".join(map(str, fib(n)))
    check = ("__import__('subprocess').run(['python3','/app/fib.py'],capture_output=True,text=True,timeout=20)"
             f".stdout.strip() == {want!r}")
    sol = f"mkdir -p /app && echo \"print({want!r})\" > /app/fib.py"
    return "fib", ins, check, sol


def t_config(r: random.Random):
    host, port, name = r.choice(WORDS), r.randint(1025, 65000), r.choice(NAMES)
    ins = (f"Create /app/config.ini (create /app if needed) in INI format with a [server] section containing host = {host}.local "
           f"and port = {port}, and a [user] section containing name = {name}.")
    check = ("(lambda c: (c.read('/app/config.ini'), "
             f"c.get('server','host') == {host + '.local'!r} and c.get('server','port') == '{port}' and c.get('user','name') == {name!r})[1])"
             "(__import__('configparser').ConfigParser())")
    sol = f"mkdir -p /app && printf '[server]\\nhost = {host}.local\\nport = {port}\\n[user]\\nname = {name}\\n' > /app/config.ini"
    return "config", ins, check, sol


def t_files(r: random.Random):
    n = r.randint(3, 6)
    ins = (f"Create the directory /app/out (create /app if needed) containing {n} files named f1.txt to f{n}.txt, "
           "where each file contains only its own number (f1.txt contains 1, and so on).")
    check = ("all(open(f'/app/out/f{i}.txt').read().strip() == str(i) for i in range(1, %d))"
             " and len(__import__('os').listdir('/app/out')) == %d" % (n + 1, n))
    sol = f"mkdir -p /app/out && for i in $(seq 1 {n}); do echo $i > /app/out/f$i.txt; done"
    return "files", ins, check, sol


def t_json(r: random.Random):
    obj = {"id": r.randint(1, 999), "name": r.choice(NAMES), "tags": r.sample(WORDS, 2), "active": r.choice([True, False])}
    ins = f"Create /app/data.json (create /app if needed) containing exactly this JSON object: {json.dumps(obj)}"
    check = f"__import__('json').load(open('/app/data.json')) == {obj!r}"
    sol = f"mkdir -p /app && echo {shlex.quote(json.dumps(obj))} > /app/data.json"
    return "json", ins, check, sol


def t_count(r: random.Random):
    lines = [" ".join(r.sample(WORDS, 3)) for _ in range(r.randint(6, 10))]
    key = r.choice(WORDS)
    n = sum(key in l.split() for l in lines)
    ins = (f"Create /app/text.txt (create /app if needed) containing these lines, one per line:\n\n" + "\n".join(lines) +
           f"\n\nThen write to /app/count.txt the number of lines that contain the word {key}. The file must contain only the number.")
    check = (f"open('/app/text.txt').read().strip().splitlines() == {lines!r} and open('/app/count.txt').read().strip() == {str(n)!r}")
    sol = f"mkdir -p /app && printf '%s\\n' {' '.join(shlex.quote(l) for l in lines)} > /app/text.txt && echo {n} > /app/count.txt"
    return "count", ins, check, sol


FAMILIES = [t_sum, t_sorted, t_reverse, t_fib, t_config, t_files, t_json, t_count]

TASK_TOML = """schema_version = "1.0"

[task]
name = "micro/{name}"
description = "Generated micro-task ({family}) for the OpenSandbox plumbing run."
keywords = ["micro", "{family}"]

[metadata]
difficulty = "easy"
category = "terminal"

[verifier]
network_mode = "public"
timeout_sec = 120.0

[agent]
network_mode = "public"
timeout_sec = 900.0

[environment]
docker_image = "{image}"
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
storage_mb = 2048
gpus = 0
"""

TEST_SH = """#!/bin/bash
# Reward is 1 when the check below holds, else 0. Never leaves reward.txt unwritten.
mkdir -p /logs/verifier
if python3 - <<'PYEOF'
import sys
try:
    ok = bool({check})
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PYEOF
then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
"""


def write_task(root: Path, idx: int, family: str, ins: str, check: str, sol: str) -> None:
    name = f"micro-{family}-{idx:03d}"
    d = root / name
    for sub in ("environment", "tests", "solution"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    (d / "task.toml").write_text(TASK_TOML.format(name=name, family=family, image=IMAGE))
    (d / "instruction.md").write_text(ins + "\n")
    # Harbor requires environment/ to exist; the Dockerfile is never built (docker_image is used).
    (d / "environment" / "Dockerfile").write_text(f"FROM {IMAGE}\n")
    (d / "tests" / "test.sh").write_text(TEST_SH.format(check=check))
    (d / "solution" / "solve.sh").write_text("#!/bin/bash\nset -e\n" + sol + "\n")
    for f in (d / "tests" / "test.sh", d / "solution" / "solve.sh"):
        f.chmod(0o755)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--count", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-validation", type=int, default=8)
    ap.add_argument("--alias", default="swe_lego", help="dataset alias used in instance_id")
    a = ap.parse_args()
    r = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(a.count):
        fam = FAMILIES[i % len(FAMILIES)]
        family, ins, check, sol = fam(r)
        write_task(a.out, i, family, ins, check, sol)
        names.append(f"micro-{family}-{i:03d}")
    # Rows in the format NemoGymDataset reads for harbor_agent (same as prepare_harbor_osb_dataset.py).
    order = names[:]
    random.Random(a.seed).shuffle(order)
    val, train = order[:a.num_validation], order[a.num_validation:]
    root = a.out.parent

    def rows(selected):
        return "".join(json.dumps({
            "instance_id": f"{a.alias}::{n}",
            "responses_create_params": {"input": [], "temperature": 1.0, "top_p": 1.0},
            "agent_ref": {"type": "responses_api_agents", "name": "harbor_agent"},
        }) + "\n" for n in selected)

    (root / "train.jsonl").write_text(rows(train))
    (root / "validation.jsonl").write_text(rows(val))
    (root / "task_names.txt").write_text("\n".join(names) + "\n")
    print(f"wrote {a.count} micro-tasks to {a.out}; {len(train)} train / {len(val)} validation rows in {root}")


if __name__ == "__main__":
    main()
