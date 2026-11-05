"""Compile verified proofs from Nemotron-Math-Proofs through the sandbox the way Gym math_formal_lean does
(its own proof-building functions, read from app.py), for rows as Gym prepare script writes them and for
rows with the statement ending in := by and the header ending in a newline.
    python3 test_sandbox.py <gym app.py> <sample proofs.jsonl> <sandbox host> [n]
"""
import ast, json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
app, samples, host = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 64
tree = ast.parse(open(app).read())
keep = {"ProofBuildConfig", "extract_code_block", "clean_formal_generation", "extract_proof_only", "build_lean4_proof", "determine_proof_status"}
mod = ast.Module(body=[x for x in tree.body if getattr(x, "name", None) in keep], type_ignores=[])
ns = {"re": re, "dataclass": dataclass, "Optional": Optional, "Dict": Dict, "Any": Any, "List": List}
exec(compile(mod, app, "exec"), ns)
rows = [json.loads(l) for l in open(samples)][:n]
def gym_row(r):  # as prepare_nemotron_math_proofs.py writes it
    stmt = r["formal_statement"]
    if not re.search(r"by\s+sorry\s*$", stmt): stmt = stmt.rstrip() + " := by sorry"
    return {"header": r["lean_header"] or "", "formal_statement": stmt}
def fixed_row(r):
    g = gym_row(r)
    return {"header": g["header"].rstrip("\n") + "\n", "formal_statement": re.sub(r"\s*sorry\s*$", "\n", g["formal_statement"])}
def compile_(code):
    req = urllib.request.Request("http://%s:6000/execute" % host, json.dumps({"generated_code": code, "language": "lean4", "timeout": 120, "max_output_characters": 2000}).encode(), {"Content-Type": "application/json"})
    t = time.time(); out = json.loads(urllib.request.urlopen(req, timeout=200).read()); out["secs"] = time.time() - t
    return out
cfg = ns["ProofBuildConfig"]()
whole = ns["ProofBuildConfig"](restate_formal_statement=False, strip_theorem_from_proof=False)
modes = [("gym rows", lambda r: ns["build_lean4_proof"](r["answer"], gym_row(r), cfg)),
         ("fixed rows", lambda r: ns["build_lean4_proof"](r["answer"], fixed_row(r), cfg)),
         ("whole answer", lambda r: ns["build_lean4_proof"](r["answer"], {"header": "", "formal_statement": ""}, whole))]
for label, build in modes[int(sys.argv[5]) if len(sys.argv) > 5 else 0:]:
    codes = [build(r) for r in rows]
    with ThreadPoolExecutor(32) as ex: outs = list(ex.map(compile_, codes))
    stat = {}
    for o in outs: s = ns["determine_proof_status"](o); stat[s] = stat.get(s, 0) + 1
    secs = sorted(o["secs"] for o in outs)
    print(label, stat, "secs p50 %.1f max %.1f" % (secs[len(secs)//2], secs[-1]))
    bad = [(c, o) for c, o in zip(codes, outs) if ns["determine_proof_status"](o) != "completed"][:2]
    for c, o in bad: print("---", label, "example code tail:", c[-300:].replace("\n", "\\n"), "\n   out:", (o["stdout"] + o["stderr"])[:500].replace("\n", "\\n"))
