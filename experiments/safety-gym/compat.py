#!/usr/bin/env python3
"""Render a bind-mounted adapter; never edit the staged NeMo-RL checkout.

The pinned Gym collector requires integer task indices. The previous local
adapter converted UUIDs to 128-bit integers, which Gym's orjson rejects.
Keep UUIDs intact and derive a stable, JSON-safe signed-63-bit task index.
"""

import argparse
import ast
import inspect
from pathlib import Path


def _ensure_legacy_gym_indices(rows):
    import hashlib
    import uuid

    maximum = (1 << 63) - 1
    tasks = {}
    owners = {}
    seen = set()
    for row in rows:
        group = row.get("_ng_group_id")
        task = row.get("_ng_task_index")
        rollout = row.get("_ng_rollout_index")
        if type(rollout) is not int or not 0 <= rollout <= maximum:
            raise ValueError("Gym rollout index must be a nonnegative signed-64-bit integer")
        if group is not None:
            canonical = uuid.UUID(group).hex
            if task is None:
                digest = hashlib.sha256(bytes.fromhex(canonical)).digest()
                task = tasks.get(canonical, int.from_bytes(digest[:8], "big") & maximum)
            if type(task) is not int or not 0 <= task <= maximum:
                raise ValueError("Gym task index must be a nonnegative signed-64-bit integer")
            if canonical in tasks and tasks[canonical] != task:
                raise ValueError("Gym prompt group has inconsistent task indices")
            tasks[canonical] = task
            if task in owners and owners[task] != canonical:
                raise ValueError("Gym task-index collision between distinct prompt groups")
            owners[task] = canonical
        elif task is None:
            index = row.get("_rowidx")
            if type(index) is not int or not row.get("_ng_rollout_id"):
                raise ValueError("Gym task requires a group UUID or explicit task and rollout identity")
            task = index - rollout
        if type(task) is not int or not 0 <= task <= maximum:
            raise ValueError("Gym task index must be a nonnegative signed-64-bit integer")
        identity = (task, rollout)
        if identity in seen:
            raise ValueError("Gym rollout batch contains duplicate task/rollout identities")
        seen.add(identity)
        row["_ng_task_index"] = task


def render_adapter(source):
    """Replace only the compatibility helper and preserve Ray actor decoration."""
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NemoGym"]
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ensure_legacy_gym_indices"]
    if len(classes) != 1 or len(helpers) > 1:
        raise ValueError("Unknown NemoGym adapter layout")
    cls = classes[0]
    if not any(isinstance(n.func if isinstance(n, ast.Call) else n, ast.Attribute)
               and (n.func if isinstance(n, ast.Call) else n).attr == "remote"
               for n in cls.decorator_list):
        raise ValueError("NemoGym must retain its Ray actor decorator")
    helper = inspect.getsource(_ensure_legacy_gym_indices)
    lines = source.splitlines(keepends=True)
    call = "        _ensure_legacy_gym_indices(nemo_gym_examples)\n"
    boundary = "        nemo_gym_result_iterator = self.rch.run_examples(\n"
    if helpers:
        node = helpers[0]
        if node.decorator_list or source.count(call) != 1:
            raise ValueError("Unknown compatibility helper or call layout")
        source = "".join(lines[:node.lineno - 1]) + helper + "".join(lines[node.end_lineno:])
    else:
        if source.count(boundary) != 1 or call in source:
            raise ValueError("Unknown NeMo Gym request boundary")
        insertion = min(n.lineno for n in cls.decorator_list) - 1
        source = "".join(lines[:insertion]) + helper + "\n\n" + "".join(lines[insertion:])
        source = source.replace(boundary, call + boundary, 1)
    compile(source, "nemo_gym.py", "exec")
    return source



def render_adaptive_grpo(source):
    """Feature-only GRPO mask and optimizer guard overlay."""
    from adaptive_runtime_overlay import render_adaptive_grpo as render
    return render(source)


def render_adaptive_collector(source):
    """Feature-only native prompt ordinal attestation overlay."""
    from adaptive_runtime_overlay import render_adaptive_collector as render
    return render(source)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("--output must differ from the staged source")
    output = render_adapter(args.source.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)
    print(f"Rendered JSON-safe Gym adapter: {args.output}")


if __name__ == "__main__":
    main()
