#!/usr/bin/env python3
"""Keep the Gym Lean rows whose statement still elaborates on the Kimina server's Lean and Mathlib.

Gym math_formal_lean compiles header + formal_statement + proof. The rows of backends/nemo-rl/lean were made
for the NeMo-Skills sandbox (Lean and Mathlib v4.12); Mathlib renames and moves lemmas between versions, so
on another version some statements no longer elaborate, and every rollout on them would score 0 whatever the
model writes. This checks header + formal_statement + "sorry" and keeps a row only if Lean reports nothing
but the `sorry` warning. Dropped rows go to <dst>.dropped.jsonl with Lean's first error.

    python3 validate_statements.py --url http://<node>:<port> --api-key <key> --max-inflight 128 \
        train.jsonl train.v434.jsonl
"""
import argparse
import asyncio
import collections
import json

from kimina_client import Snippet, SnippetStatus, StreamingChecker


def first_error(result) -> str:
    if result.error:
        return result.error[:500]
    for m in (result.response or {}).get("messages") or []:
        if m.get("severity") == "error":
            return str(m.get("data", ""))[:500]
    return ""


async def run(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in open(args.src)]
    # Gym appends the proof after the statement, so a statement must end with ":= by" (fix_rows.py makes the
    # NeMo-Skills rows so); one that does not is dropped unchecked.
    wellformed = [r["formal_statement"].rstrip().endswith(":= by") for r in rows]
    async with StreamingChecker(
        api_url=args.url, api_key=args.api_key, max_inflight=args.max_inflight, timeout=args.timeout
    ) as checker:
        results = await checker.check(
            Snippet(id=str(i), code=r["header"] + r["formal_statement"] + "  sorry\n")
            for i, (r, ok) in enumerate(zip(rows, wellformed)) if ok
        )
    checked = iter(results.results)
    status, kept = [], 0
    with open(args.dst, "w") as f, open(args.dst + ".dropped.jsonl", "w") as d:
        for row, ok in zip(rows, wellformed):
            if not ok:
                status.append("malformed_statement")
                d.write(json.dumps({"name": row.get("name"), "status": "malformed_statement", "error": row["formal_statement"][-80:]}, ensure_ascii=False) + "\n")
                continue
            res = next(checked)
            s = res.analyze().status
            status.append(s.value)
            if s == SnippetStatus.sorry:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            else:
                d.write(json.dumps({"name": row.get("name"), "status": s.value, "error": first_error(res)}, ensure_ascii=False) + "\n")
    counts = collections.Counter(status)
    print(json.dumps({"src": args.src, "rows": len(rows), "kept": kept, "status": dict(counts),
                      "unchecked": checker.stats["unchecked"]}))
    if checker.stats["unchecked"]:
        raise SystemExit(f"{checker.stats['unchecked']} statements were not checked: rerun")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--url", required=True)
    ap.add_argument("--api-key")
    ap.add_argument("--max-inflight", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=120)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
