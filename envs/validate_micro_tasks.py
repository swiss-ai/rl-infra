#!/usr/bin/env python3
"""Check the micro-tasks on real OpenSandbox pods: the tests must give reward 0 on the untouched
image and reward 1 after the reference solution (the nop and oracle checks of Harbor).

    OPENSANDBOX_API_KEY=$(tr -d '\\n' < ~/.opensandbox_token) \\
        python3 validate_micro_tasks.py <dir>/tasks [--per-family 1]

One sandbox per task, deleted afterwards. Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

API = os.environ.get("OPENSANDBOX_DOMAIN", "https://opensandbox.breithorn.svc.cscs.ch").rstrip("/") + "/v1"
KEY = os.environ["OPENSANDBOX_API_KEY"]


def call(method, url, body=None, headers=None, stream=False):
    h = {"OPEN-SANDBOX-API-KEY": KEY, "Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, method=method, headers=h, data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read().decode()
    return raw if stream else (json.loads(raw) if raw else {})


def run(ep, cmd):
    raw = call("POST", f"https://{ep['endpoint']}/command", {"command": cmd}, ep["headers"], stream=True)
    out = ""
    for line in raw.splitlines():
        line = line.strip().removeprefix("data:").strip()
        if line.startswith("{"):
            ev = json.loads(line)
            if ev.get("type") in ("stdout", "stderr"):
                out += ev.get("text", "")
    return out


def reward(ep):
    m = re.findall(r"^[01]$", run(ep, "cat /logs/verifier/reward.txt").strip(), re.M)
    return int(m[-1]) if m else None


def check(task: Path) -> bool:
    toml = (task / "task.toml").read_text()
    image = re.search(r'docker_image = "([^"]+)"', toml).group(1)
    sb = call("POST", f"{API}/sandboxes", {
        "image": {"uri": image}, "timeout": 300, "secureAccess": True,
        "platform": {"os": "linux", "arch": "amd64"},
        "resourceLimits": {"cpu": "1", "memory": "1Gi"}, "entrypoint": ["tail", "-f", "/dev/null"]})
    sid = sb["id"]
    try:
        ep = call("GET", f"{API}/sandboxes/{sid}/endpoints/44772", headers={})
        # wait for execd
        for _ in range(30):
            try:
                if "ok" in run(ep, "echo ok"):
                    break
            except Exception:
                pass
        put = lambda path, f: run(ep, f"mkdir -p $(dirname {path}) && echo {base64.b64encode(f.read_bytes()).decode()} | base64 -d > {path} && chmod +x {path}")
        put("/tests/test.sh", task / "tests" / "test.sh")
        put("/solution/solve.sh", task / "solution" / "solve.sh")
        run(ep, "bash /tests/test.sh")
        before = reward(ep)
        run(ep, "bash /solution/solve.sh && bash /tests/test.sh")
        after = reward(ep)
        ok = (before, after) == (0, 1)
        print(f"{'OK  ' if ok else 'FAIL'} {task.name}: before={before} after={after}")
        return ok
    finally:
        try:
            call("DELETE", f"{API}/sandboxes/{sid}")
        except Exception as e:
            print(f"  warning: could not delete {sid}: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", type=Path)
    ap.add_argument("--per-family", type=int, default=1)
    a = ap.parse_args()
    seen, picked = {}, []
    for t in sorted(p for p in a.tasks.iterdir() if (p / "task.toml").exists()):
        fam = t.name.split("-")[1]
        if seen.get(fam, 0) < a.per_family:
            seen[fam] = seen.get(fam, 0) + 1
            picked.append(t)
    results = [check(t) for t in picked]
    print("ALL OK" if all(results) else "SOME FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
