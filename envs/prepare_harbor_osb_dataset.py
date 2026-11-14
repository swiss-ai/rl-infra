#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a NeMo-RL train/validation split for harbor_agent on OpenSandbox.

OpenSandbox sibling of prepare_harbor_dataset.py. That script downloaded a
task subset, built a shared `.sif` and rewrote every task.toml to point at it.
Here the task tree is an input, not an output: it already exists on the
training cluster, and every task.toml already names a prebuilt image the
OpenSandbox cluster can pull:

    <tasks-dir>/<task_name>/task.toml         [environment].docker_image = "<registry>/<image>:<tag>"
    <tasks-dir>/<task_name>/instruction.md
    <tasks-dir>/<task_name>/tests/            verifier, uploaded into the sandbox by Harbor

So this script only validates the tree and writes the rows. A row carries
only `instance_id`, formatted as `<alias>::<task_name>`; the alias resolves
through the recipe's `harbor_datasets` map to `<tasks-dir>`, and Harbor loads
the task from disk. `agent_ref` is what NeMo-RL routes on.

Tasks without `docker_image` are skipped, not fatal, so a partially built
image set can already be trained on; the counts are reported.

Usage:
    python3.12 envs/prepare_harbor_osb_dataset.py \\
        --tasks-dir "$HARBOR_TASKS_DIR" \\
        --output-dir "$HARBOR_ROOT" \\
        --limit 64 --num-validation 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tomllib
from pathlib import Path


def read_docker_image(task_toml: Path) -> str | None:
    """Return the task's `[environment].docker_image`, or None if unset."""
    with task_toml.open("rb") as handle:
        config = tomllib.load(handle)
    image = (config.get("environment") or {}).get("docker_image")
    return image or None


def to_gym_row(
    instance_id: str, agent_name: str, temperature: float, top_p: float
) -> dict:
    """Build one harbor_agent row.

    Args:
        instance_id: "<alias>::<task_name>".
        agent_name: Gym server instance name; must match a top-level key in the
            merged Gym config.
        temperature: Sampling temperature for the policy calls.
        top_p: Nucleus sampling cutoff for the policy calls.

    Returns:
        A row in the shape HarborRunRequest expects, plus the agent_ref NeMo-RL
        requires.
    """
    return {
        "instance_id": instance_id,
        "responses_create_params": {
            # Harbor builds the real prompt from the task's instruction.md, so
            # the seed input is empty.
            "input": [],
            # Must match policy.generation.{temperature,top_p} in the recipe:
            # NeMo-RL's vLLM server asserts every inbound request carries the
            # on-policy sampling params.
            "temperature": temperature,
            "top_p": top_p,
        },
        "agent_ref": {"type": "responses_api_agents", "name": agent_name},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="Directory of Harbor tasks; the recipe's "
        "harbor_datasets.<alias>.local_dataset_path must point here",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alias",
        default="verified",
        help="Left half of every instance_id; must match the recipe's harbor_datasets key",
    )
    parser.add_argument("--agent-name", default="harbor_agent")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--num-validation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    tasks_dir = args.tasks_dir.resolve()
    if not tasks_dir.is_dir():
        sys.exit(f"Not a directory: {tasks_dir}")

    task_names = []
    missing_image = []
    for path in sorted(tasks_dir.iterdir()):
        task_toml = path / "task.toml"
        if not task_toml.is_file():
            continue
        if read_docker_image(task_toml) is None:
            missing_image.append(path.name)
            continue
        task_names.append(path.name)

    print(
        f"{len(task_names)} tasks with a docker_image, "
        f"{len(missing_image)} without (skipped) under {tasks_dir}",
        flush=True,
    )
    if missing_image:
        print(f"  e.g. skipped: {', '.join(missing_image[:5])}", flush=True)
    # No environment/ check: harbor >=0.20 requires that directory only when
    # the task has no docker_image (Task.is_valid_dir), and every task that
    # reaches here has one. Older harbor demanded it unconditionally, so this
    # used to warn that Harbor would silently resolve the dataset to zero
    # tasks; that no longer happens.
    if not task_names:
        sys.exit("No usable tasks: every task.toml must set [environment].docker_image")

    random.Random(args.seed).shuffle(task_names)
    selected = task_names[: args.limit]
    if len(selected) < args.limit:
        print(
            f"WARNING: only {len(selected)} tasks available (--limit {args.limit})",
            flush=True,
        )
    if args.num_validation >= len(selected):
        sys.exit(
            f"--num-validation ({args.num_validation}) must be < selected tasks ({len(selected)})"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = (
        ("validation", selected[: args.num_validation]),
        ("train", selected[args.num_validation :]),
    )
    for split_name, subset in splits:
        path = args.output_dir / f"{split_name}.jsonl"
        with path.open("w") as handle:
            for task_name in subset:
                row = to_gym_row(
                    f"{args.alias}::{task_name}",
                    args.agent_name,
                    args.temperature,
                    args.top_p,
                )
                handle.write(json.dumps(row) + "\n")
        print(f"Wrote {len(subset)} rows -> {path}", flush=True)

    task_list = args.output_dir / "task_names.txt"
    task_list.write_text("".join(f"{name}\n" for name in selected))
    print(f"Wrote {len(selected)} task names -> {task_list}", flush=True)
    print(
        f"\nLaunch with HARBOR_TASKS_DIR={tasks_dir}: the recipe resolves "
        f"harbor_datasets.{args.alias}.local_dataset_path from it.",
        flush=True,
    )


if __name__ == "__main__":
    main()
