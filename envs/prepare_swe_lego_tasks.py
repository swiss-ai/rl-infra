#!/usr/bin/env python3
"""Turn Ambroise's validated SWE-Lego Harbor tasks into a HARBOR_ROOT for run-harbor-osb.sh.

The source (on Alps: /capstor/store/cscs/swissai/infra01/users/aborbely/swe_lego_v05/tasks) holds one
directory per task, <src>/<task>/task/<task>/ with task.toml, instruction.md, tests/ and solution/, and the
task's oracle/nop check results in <src>/<task>/out/results/*.jsonl. This script:

  - keeps the tasks whose reference solution scored 1 ("oracle") and whose empty run scored 0 ("nop");
  - copies each to <out>/tasks/<task>/ (the flat layout envs/harbor-osb.yaml expects);
  - adds environment/Dockerfile ("FROM <docker_image>"): the Harbor version the Gym checkout installs
    only accepts a task directory with an environment/ directory, and finds no task at all without it
    ("Either datasets or tasks must be provided", KISTI job 1170968);
  - writes <out>/train.jsonl, <out>/validation.jsonl and <out>/task_names.txt, rows `swe_lego::<task>`.

    python3 prepare_swe_lego_tasks.py <src> <out> [--num-validation 16] [--limit N] [--seed 0]

Then point HARBOR_ROOT at <out>. The images are in the public rl-registry project swe_lego_v05_1, so the
placeholder registry credentials work. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path


def checks(task_dir: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for f in (task_dir / "out" / "results").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[row.get("check")] = row.get("reward")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--num-validation", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="keep at most N tasks (after shuffling)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alias", default="swe_lego", help="dataset alias used in instance_id")
    a = ap.parse_args()

    kept, skipped = [], {"no task.toml": 0, "no docker_image": 0, "oracle != 1": 0, "nop != 0": 0}
    for d in sorted(p for p in a.src.iterdir() if p.is_dir()):
        task = d / "task" / d.name
        if not (task / "task.toml").exists():
            skipped["no task.toml"] += 1
            continue
        if not re.search(r'^docker_image = "[^"]+"', (task / "task.toml").read_text(), re.M):
            skipped["no docker_image"] += 1
            continue
        c = checks(d)
        if c.get("oracle") != 1.0:
            skipped["oracle != 1"] += 1
            continue
        if c.get("nop") not in (0, 0.0):
            skipped["nop != 0"] += 1
            continue
        kept.append(d.name)

    random.Random(a.seed).shuffle(kept)
    kept = kept[: a.limit] if a.limit else kept
    val, train = sorted(kept[: a.num_validation]), sorted(kept[a.num_validation:])

    tasks = a.out / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    for name in kept:
        dst = tasks / name
        shutil.copytree(a.src / name / "task" / name, dst, dirs_exist_ok=True)
        image = re.search(r'^docker_image = "([^"]+)"', (dst / "task.toml").read_text(), re.M).group(1)
        (dst / "environment").mkdir(exist_ok=True)
        (dst / "environment" / "Dockerfile").write_text(f"FROM {image}\n")

    def rows(names):
        return "".join(json.dumps({
            "instance_id": f"{a.alias}::{n}",
            "responses_create_params": {"input": [], "temperature": 1.0, "top_p": 1.0},
            "agent_ref": {"type": "responses_api_agents", "name": "harbor_agent"},
        }) + "\n" for n in names)

    (a.out / "train.jsonl").write_text(rows(train))
    (a.out / "validation.jsonl").write_text(rows(val))
    (a.out / "task_names.txt").write_text("\n".join(sorted(kept)) + "\n")
    print(f"{len(kept)} tasks ({len(train)} train, {len(val)} validation) in {a.out}; skipped {skipped}")


if __name__ == "__main__":
    main()
