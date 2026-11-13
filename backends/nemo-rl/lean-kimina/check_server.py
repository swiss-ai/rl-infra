#!/usr/bin/env python3
"""Checks of a running Kimina server before and after the data are validated.

    python3 check_server.py controls <url> <key> <header>
        Four snippets under the dataset's header whose verdicts are known (two valid proofs, a sorry, an error).
        Exits 1 on any other verdict: a server whose REPLs did not import the header would fail them all.
    python3 check_server.py samples <url> <key> <gym app.py> <nemotron_sample_proofs.jsonl> [n]
        Verified proofs from Nemotron-Math-Proofs (verified on Lean/Mathlib v4.12), built into code the way Gym
        math_formal_lean does (its own functions, read from app.py, as backends/nemo-rl/lean/test_sandbox.py
        does) for rows as fix_rows.py leaves them. Prints how many still pass on this server's Lean version.
"""
import ast
import asyncio
import collections
import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from kimina_client import Snippet, SnippetStatus, StreamingChecker

CONTROLS = [
    ("valid-norm_num", "theorem kimina_control_1 : (2:ℕ) + 2 = 4 := by norm_num\n", SnippetStatus.valid),
    ("valid-nlinarith", "theorem kimina_control_2 (a b : ℝ) : 2*a*b ≤ a^2 + b^2 := by\n  nlinarith [sq_nonneg (a - b)]\n", SnippetStatus.valid),
    ("sorry", "theorem kimina_control_3 : (1:ℕ) = 1 := by sorry\n", SnippetStatus.sorry),
    ("error", "theorem kimina_control_4 : (1:ℕ) = 2 := by norm_num\n", SnippetStatus.lean_error),
]


async def check(url: str, key: str, snippets: list[Snippet], inflight: int = 64, timeout: int = 120):
    async with StreamingChecker(api_url=url, api_key=key, max_inflight=inflight, timeout=timeout) as c:
        return (await c.check(snippets)).results


def controls(url: str, key: str, header: str) -> None:
    results = asyncio.run(check(url, key, [Snippet(id=i, code=header + "\n\n" + code) for i, code, _ in CONTROLS]))
    bad = {r.id: r.analyze().status.value for r, (_, _, want) in zip(results, CONTROLS) if r.analyze().status != want}
    print("controls:", "all as expected" if not bad else f"MISMATCH {bad}")
    sys.exit(1 if bad else 0)


def gym_functions(app_py: str) -> dict:
    keep = {"ProofBuildConfig", "extract_code_block", "clean_formal_generation", "extract_proof_only", "build_lean4_proof"}
    tree = ast.parse(open(app_py).read())
    mod = ast.Module(body=[x for x in tree.body if getattr(x, "name", None) in keep], type_ignores=[])
    ns = {"re": re, "dataclass": dataclass, "Optional": Optional, "Dict": Dict, "Any": Any, "List": List}
    exec(compile(mod, app_py, "exec"), ns)
    return ns


def samples(url: str, key: str, app_py: str, path: str, n: int) -> None:
    ns = gym_functions(app_py)
    cfg = ns["ProofBuildConfig"]()
    rows = [json.loads(line) for line in open(path)][:n]
    codes = []
    for r in rows:  # the row as prepare_nemotron_math_proofs.py writes it, then as fix_rows.py leaves it
        stmt = r["formal_statement"]
        if not re.search(r"by\s+sorry\s*$", stmt):
            stmt = stmt.rstrip() + " := by sorry"
        row = {"header": (r["lean_header"] or "").rstrip("\n") + "\n", "formal_statement": re.sub(r"\s*sorry\s*$", "\n", stmt)}
        codes.append(ns["build_lean4_proof"](r["answer"], row, cfg))
    results = asyncio.run(check(url, key, [Snippet(id=str(i), code=c) for i, c in enumerate(codes)], timeout=60))
    counts = collections.Counter(r.analyze().status.value for r in results)
    print(json.dumps({"sample_proofs": len(rows), "status": dict(counts)}))


if __name__ == "__main__":
    if sys.argv[1] == "controls":
        controls(*sys.argv[2:5])
    elif sys.argv[1] == "samples":
        samples(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]) if len(sys.argv) > 6 else 400)
    else:
        sys.exit(__doc__)
