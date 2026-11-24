#!/usr/bin/env python3
"""Keep the copy of scratch_adam.py inside Megatron patch 0007 equal to this folder's scratch_adam.py.

  python3 patch_tool.py           rewrite the patch's scratch_adam.py section from ./scratch_adam.py
  python3 patch_tool.py --check   exit 1 when they differ (stage2_test.py runs this check)
"""
import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = HERE.parents[2] / "apertus2" / "patches" / "megatron" / "0007-cpu-optimizer-state-dir-scratch-adam.patch"
TARGET = "megatron/core/optimizer/cpu_offloading/scratch_adam.py"


def _section(lines):
    """(start, end) line indexes of the new-file diff of TARGET in the patch."""
    start = next(i for i, line in enumerate(lines) if line.startswith(f"diff --git a/{TARGET} "))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("diff --git ")), len(lines))
    return start, end


def extract(patch_text):
    lines = patch_text.splitlines(keepends=True)
    start, end = _section(lines)
    hunk = next(i for i in range(start, end) if lines[i].startswith("@@ "))
    return "".join(line[1:] for line in lines[hunk + 1:end] if line.startswith("+"))


def render(patch_text, source_text):
    lines = patch_text.splitlines(keepends=True)
    start, end = _section(lines)
    blob = hashlib.sha1(b"blob %d\0" % len(source_text.encode()) + source_text.encode()).hexdigest()
    body = source_text.splitlines(keepends=True)
    section = [f"diff --git a/{TARGET} b/{TARGET}\n", "new file mode 100644\n", f"index 0000000..{blob[:9]}\n",
               "--- /dev/null\n", f"+++ b/{TARGET}\n", f"@@ -0,0 +1,{len(body)} @@\n"] + ["+" + line for line in body]
    return "".join(lines[:start] + section + lines[end:])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    source = (HERE / "scratch_adam.py").read_text()
    patch = PATCH.read_text()
    if args.check:
        same = extract(patch) == source
        print("patch 0007 is up to date" if same else f"{PATCH.name}: its scratch_adam.py differs; run patch_tool.py")
        return 0 if same else 1
    PATCH.write_text(render(patch, source))
    print(f"updated {PATCH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
