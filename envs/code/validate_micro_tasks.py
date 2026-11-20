#!/usr/bin/env python3
"""Check Harbor tasks on real OpenSandbox pods: the tests must give reward 0 on the untouched image
and reward 1 after the reference solution (Harbor's nop and oracle checks). From Robert's branch
harbor-osb, changed so that the API key is read from a file and goes only to the API, never to a pod.

    python3 validate_micro_tasks.py <dir>/tasks --api-key-file ~/.opensandbox_token [--per-family 1]

One pod per task, labelled with this user and a run id, deleted afterwards. Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

DOMAIN = "https://opensandbox-gvisor.breithorn.svc.cscs.ch"
RUN = f"task-check-{uuid.uuid4().hex[:8]}"


def call(method, url, headers, body=None, stream=False):
    h = {"Content-Type": "application/json", **headers}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, method=method, headers=h, data=data)
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read().decode()
    return raw if stream else (json.loads(raw) if raw else {})


def run(ep, cmd):
    # Only the pod's own headers: model code runs in there and must never see the API key.
    raw = call("POST", f"https://{ep['endpoint']}/command", ep.get("headers") or {}, {"command": cmd}, stream=True)
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


def check(task: Path, api: str, key: dict) -> bool:
    toml = (task / "task.toml").read_text()
    image = re.search(r'docker_image = "([^"]+)"', toml).group(1)
    sb = call("POST", f"{api}/sandboxes", key, {
        "image": {"uri": image}, "timeout": 300, "secureAccess": True,
        "platform": {"os": "linux", "arch": "amd64"},
        "resourceLimits": {"cpu": "1", "memory": "1Gi"}, "entrypoint": ["tail", "-f", "/dev/null"],
        "metadata": {"nemo-gym.nvidia.com/user": os.environ.get("USER", "unknown"),
                     "nemo-gym.nvidia.com/workload": "harbor-task-check",
                     "nemo-gym.nvidia.com/run": RUN}})
    sid = sb["id"]
    try:
        ep = call("GET", f"{api}/sandboxes/{sid}/endpoints/44772", key)
        for _ in range(30):  # wait for the pod's exec daemon
            try:
                if "ok" in run(ep, "echo ok"):
                    break
            except Exception:
                time.sleep(2)
        put = lambda path, f: run(ep, f"mkdir -p $(dirname {path}) && echo {base64.b64encode(f.read_bytes()).decode()} | base64 -d > {path} && chmod +x {path}")
        put("/tests/test.sh", task / "tests" / "test.sh")
        put("/solution/solve.sh", task / "solution" / "solve.sh")
        run(ep, "bash /tests/test.sh")
        before = reward(ep)
        run(ep, "bash /solution/solve.sh && bash /tests/test.sh")
        after = reward(ep)
        ok = (before, after) == (0, 1)
        print(f"{'OK  ' if ok else 'FAIL'} {task.name}: before={before} after={after}", flush=True)
        return ok
    finally:
        try:
            call("DELETE", f"{api}/sandboxes/{sid}", key)
        except Exception as e:
            print(f"  warning: could not delete {sid} ({type(e).__name__})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", type=Path)
    ap.add_argument("--api-key-file", type=Path, required=True)
    ap.add_argument("--domain", default=os.environ.get("OPENSANDBOX_DOMAIN", DOMAIN))
    ap.add_argument("--per-family", type=int, default=1)
    a = ap.parse_args()
    api = a.domain.rstrip("/") + "/v1"
    key = {"OPEN-SANDBOX-API-KEY": a.api_key_file.read_text().strip()}
    seen, picked = {}, []
    for t in sorted(p for p in a.tasks.iterdir() if (p / "task.toml").exists()):
        fam = t.name.split("-")[1]
        if seen.get(fam, 0) < a.per_family:
            seen[fam] = seen.get(fam, 0) + 1
            picked.append(t)
    print(f"checking {len(picked)} tasks, pods labelled run={RUN}", flush=True)
    results = [check(t, api, key) for t in picked]
    print("ALL OK" if all(results) else "SOME FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
