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

"""Extract a harbor_verified_tasks store into a flat Harbor task tree.

Input is the shared verified-task store, one directory per validation
campaign, with the tasks that campaign produced shipped as tarballs:

    <verified-dir>/<campaign>/tasks/<task_name>.tar   the task, as a tarball
    <verified-dir>/<campaign>/tasks.csv               per-task verdict + image
    <verified-dir>/<campaign>/{attempts,logs,...}     campaign bookkeeping

Harbor wants the flat, extracted layout that `--local-dataset-path` resolves
against:

    <out-dir>/<campaign>__<task_name>/task.toml

So this script discovers the tarballs, filters them by the verdict the
campaign recorded, and extracts each one into `<out-dir>`. Each tarball holds
a single top-level directory named after the task; that level is flattened
away so `task.toml` lands directly under the task directory.

Task directories are prefixed with the campaign name. The campaigns use
disjoint task names today, but nothing enforces that across campaigns, and a
silent overwrite would swap one task's tests under another's instruction.

Selection is deliberately conservative: only tasks the campaign recorded a
`keep` verdict for are taken, since a `drop` failed its own solution-patch
verification and an `unverified` was never checked at all -- neither can be
scored reliably. `--include-unverified` / `--include-dropped` opt back in.
A campaign with no `tasks.csv` has no verdicts to filter on and is skipped
unless `--include-unverified` is passed.

There is no registry join here: the campaign already pushed both the agent
and the verifier image and wrote the digest-pinned `docker_image` into the
task.toml inside the tarball.

Extraction is incremental. Each task directory keeps a `.source.json` stamp
naming the tarball it came from with that tarball's size and mtime, so a
re-run only re-extracts tasks whose tarball actually changed. `--force`
re-extracts everything.

Usage:
    python3.12 envs/collect_harbor_verified_tasks.py \\
        --verified-dir <store>/harbor_verified_tasks \\
        --out-dir "$HARBOR_TASKS_DIR"
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

STAMP_NAME = ".source.json"
KEEP_VERDICT = "keep"
DROP_VERDICT = "drop"


@dataclass(frozen=True)
class TaskTar:
    """One tarball selected for extraction."""

    name: str
    """Destination directory name, `<campaign>__<task>`."""

    tar_path: Path


def read_verdicts(campaign_dir: Path) -> dict[str, str] | None:
    """Return `{<task>.tar: verdict}` from the campaign's tasks.csv.

    Returns None when the campaign has no readable tasks.csv, which is the
    caller's signal that there is no verdict to filter on -- distinct from a
    csv that simply recorded no verdict for a task.
    """
    csv_path = campaign_dir / "tasks.csv"
    if not csv_path.is_file():
        return None
    try:
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None
    return {
        row["task"]: (row.get("verdict") or "").strip()
        for row in rows
        if row.get("task")
    }


def select_campaign(
    campaign_dir: Path,
    include_unverified: bool,
    include_dropped: bool,
) -> tuple[list[TaskTar], dict[str, int]]:
    """Pick the tarballs to extract from one campaign directory.

    Returns the selection plus a per-reason count of everything rejected, so
    the caller can report why a campaign contributed fewer tasks than it holds.
    """
    tars = sorted((campaign_dir / "tasks").glob("*.tar"))
    rejected = {DROP_VERDICT: 0, "unverified": 0}
    if not tars:
        return [], rejected

    verdicts = read_verdicts(campaign_dir)
    selected: list[TaskTar] = []
    for tar_path in tars:
        # A missing tasks.csv and a blank verdict both mean "never validated".
        verdict = (verdicts or {}).get(tar_path.name, "") or "unverified"
        if verdict == KEEP_VERDICT:
            pass
        elif verdict == DROP_VERDICT:
            if not include_dropped:
                rejected[DROP_VERDICT] += 1
                continue
        elif not include_unverified:
            rejected["unverified"] += 1
            continue
        selected.append(TaskTar(f"{campaign_dir.name}__{tar_path.stem}", tar_path))
    return selected, rejected


def discover(
    verified_dir: Path,
    include_unverified: bool,
    include_dropped: bool,
) -> list[TaskTar]:
    """Collect the tarballs to extract under `verified_dir`.

    Accepts either the store root (every subdirectory is a campaign) or a
    single campaign directory, so a one-campaign set can be built without
    staging it under a throwaway parent.
    """
    campaigns = (
        [verified_dir]
        if (verified_dir / "tasks").is_dir()
        else [entry for entry in sorted(verified_dir.iterdir()) if entry.is_dir()]
    )

    selected: list[TaskTar] = []
    for campaign_dir in campaigns:
        tasks, rejected = select_campaign(
            campaign_dir, include_unverified, include_dropped
        )
        if not tasks and not any(rejected.values()):
            continue
        reasons = ", ".join(
            f"{count} {name}" for name, count in rejected.items() if count
        )
        print(
            f"  {campaign_dir.name}: {len(tasks)} selected"
            + (f" ({reasons} rejected)" if reasons else ""),
            flush=True,
        )
        selected.extend(tasks)
    return sorted(selected, key=lambda task: task.name)


def stamp_for(tar_path: Path) -> dict:
    """Identify a tarball well enough to tell an unchanged one from a new one."""
    info = tar_path.stat()
    return {
        "tar": str(tar_path),
        "size": info.st_size,
        "mtime": info.st_mtime,
    }


def is_current(destination: Path, stamp: dict) -> bool:
    """Return whether `destination` already holds this exact tarball's contents."""
    stamp_path = destination / STAMP_NAME
    if not (destination / "task.toml").is_file() or not stamp_path.is_file():
        return False
    try:
        return json.loads(stamp_path.read_text()) == stamp
    except (OSError, ValueError):
        return False


def remove(path: Path) -> None:
    """Delete `path` whatever it is, including a dangling symlink."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def extract(task: TaskTar, out_dir: Path, stamp: dict) -> None:
    """Extract one tarball into `<out_dir>/<task.name>`, flattening its root.

    Extraction goes to a sibling staging directory and is moved into place
    only once complete, so an interrupted run can never leave a half-extracted
    task that the dataset step would then happily pick up.
    """
    destination = out_dir / task.name
    staging = out_dir / f".{task.name}.partial"
    remove(staging)
    staging.mkdir(parents=True)
    try:
        with tarfile.open(task.tar_path) as archive:
            # `data` rejects absolute paths, parent traversal, links escaping
            # the destination, and device nodes.
            archive.extractall(staging, filter="data")

        entries = list(staging.iterdir())
        root = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
        if not (root / "task.toml").is_file():
            raise ValueError(f"no task.toml in {task.tar_path}")
        (root / STAMP_NAME).write_text(json.dumps(stamp) + "\n")

        remove(destination)
        root.rename(destination)
    finally:
        remove(staging)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--verified-dir",
        type=Path,
        required=True,
        help="harbor_verified_tasks store root, or a single campaign directory.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Keep at most this many tasks (0 = no limit). Applied after "
        "sorting by name, so the subset is stable across runs.",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="Also take tasks whose campaign recorded no verdict. Off by "
        "default: such a task was never run against its own solution patch.",
    )
    parser.add_argument(
        "--include-dropped",
        action="store_true",
        help="Also take tasks the campaign marked drop. Off by default: a "
        "dropped task failed its own solution-patch verification, so it "
        "cannot be scored reliably.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract every task, even one whose tarball is unchanged.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=16,
        help="Tarballs to extract concurrently.",
    )
    args = parser.parse_args()

    verified_dir = args.verified_dir.resolve()
    if not verified_dir.is_dir():
        sys.exit(f"Not a directory: {verified_dir}")

    print(f"Scanning {verified_dir}", flush=True)
    selected = discover(verified_dir, args.include_unverified, args.include_dropped)
    if not selected:
        sys.exit("No usable tasks: nothing to extract")
    print(f"{len(selected)} tasks selected under {verified_dir}", flush=True)

    if args.limit:
        selected = selected[: args.limit]
        print(f"Limiting to {len(selected)} tasks", flush=True)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Tasks that vanished upstream, or that a narrower --limit excluded, would
    # otherwise linger and still be picked up by the dataset step, so the tree
    # is pruned to exactly this selection.
    keep_names = {task.name for task in selected}
    for stale in sorted(out_dir.iterdir()):
        if stale.name not in keep_names:
            remove(stale)
            print(f"  removed stale task: {stale.name}", flush=True)

    pending = []
    for task in selected:
        stamp = stamp_for(task.tar_path)
        if args.force or not is_current(out_dir / task.name, stamp):
            pending.append((task, stamp))
    print(
        f"{len(pending)} to extract, {len(selected) - len(pending)} already current",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        # list() so an exception in any worker surfaces here rather than being
        # swallowed into a silently incomplete tree.
        list(pool.map(lambda item: extract(item[0], out_dir, item[1]), pending))

    print(f"Wrote {len(selected)} tasks to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
