#!/usr/bin/env python3
"""Copy a Lean toolchain and a project's compiled packages into node memory, in parallel.

    python3 stage_lean.py --toolchain <elan>/toolchains/<tc> --project <ProofNet-Verified> \
        --dst /dev/shm/$USER-kimina-$SLURM_JOB_ID --jobs 64 > staged.json

Lean opens ~34k files to `import Mathlib`. From Lustre that costs ~50 s per cold import alone
(measured on Clariden, 2026-11-05) and much more with many imports at once; a serial `cp -a` into
/dev/shm took 723 s. Here many threads copy at once, and files `lean` never reads on import are
skipped: `.ilean` (language-server references), `.trace` and the `*.hash` files (lake's build
bookkeeping). The `.ir.sig` files ARE needed: without them `import` fails with "Could not find
native implementation of external declaration" (checked on Batteries, Lean v4.34.0), and a REPL
then elaborates every proof in an environment without the imports, so all proofs fail fast.
Prints JSON: the LEAN_PATH and LEAN_SYSROOT to use, files, bytes, seconds.
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import time
from pathlib import Path

SKIP_SUFFIXES = (".ilean", ".trace", ".hash")


def wanted(name: str) -> bool:
    return not name.endswith(SKIP_SUFFIXES)


def plan(src: Path, dst: Path, filtered: bool) -> list[tuple[Path, Path]]:
    pairs = []
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        for f in files:
            if not filtered or wanted(f):
                pairs.append((Path(root) / f, dst / rel / f))
    return pairs


def copy(pair: tuple[Path, Path]) -> int:
    s, d = pair
    d.parent.mkdir(parents=True, exist_ok=True)
    if s.is_symlink():
        os.symlink(os.readlink(s), d)
        return 0
    shutil.copyfile(s, d)
    shutil.copymode(s, d)
    return d.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolchain", type=Path, required=True)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--all-files", action="store_true", help="copy build bookkeeping too")
    args = ap.parse_args()

    t0 = time.monotonic()
    pairs = plan(args.toolchain, args.dst / "toolchain", filtered=False)
    lean_path = []
    libs = sorted(args.project.glob(".lake/packages/*/.lake/build/lib/lean"))
    own = args.project / ".lake/build/lib/lean"
    if own.is_dir():
        libs.append(own)
    for lib in libs:
        name = lib.parents[3].name if lib != own else "_project"
        dst = args.dst / "packages" / name
        pairs += plan(lib, dst, filtered=not args.all_files)
        lean_path.append(str(dst))
    t_plan = time.monotonic() - t0
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        nbytes = sum(pool.map(copy, pairs, chunksize=64))
    print(
        json.dumps(
            {
                "LEAN_PATH": ":".join(lean_path),
                "LEAN_SYSROOT": str(args.dst / "toolchain"),
                "files": len(pairs),
                "gib": round(nbytes / 2**30, 2),
                "plan_s": round(t_plan, 1),
                "seconds": round(time.monotonic() - t0, 1),
                "jobs": args.jobs,
            }
        )
    )


if __name__ == "__main__":
    main()
