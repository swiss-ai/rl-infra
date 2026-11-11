#!/usr/bin/env python3
"""Why do OpenSandbox pods stay Pending? Ask the API instead of kubectl.

Job 5317993 (SWE-Lego, 4 CPU / 4 GiB / 10 GiB per pod): all 512 creates ended in
HTTP 504 KUBERNETES::POD_READY_TIMEOUT, "Last state: Pending" after 600 s. The server
deletes a sandbox when its ready wait fails, so its state has to be read while it waits.

Creates one sandbox per case, at the same time, then polls GET /v1/sandboxes/{id} and the
Kubernetes events (GET /v1/sandboxes/{id}/diagnostics/events) until it runs or the wait ends,
and deletes it. The cases tell the two causes apart:
  small      python:3.12-slim, 1 CPU / 1024 MiB / 2 GiB (as the micro tasks, which run)
  swe-small  the SWE-Lego image, the micro tasks' resources        -> does the image start?
  swe-full   the SWE-Lego image, the task's 4 CPU / 4096 MiB / 10 GiB -> do the resources fit?
swe-small runs, swe-full stays Pending: scheduling/quota. Both stay Pending: the image (pull).

Standard library only; the API key is read from a file and never printed.
  python3 experiments/osb-reach/pending.py [--task-toml $SCRATCH/osb/swe_lego_v05/tasks/<task>/task.toml]
Env: OPENSANDBOX_API_KEY_FILE (default ~/.opensandbox_token), OPENSANDBOX_DOMAIN
(default https://opensandbox-gvisor.breithorn.svc.cscs.ch), WAIT_S (default 900).
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

SWE_DEFAULT_TASK = "adamchainz__time-machine-298"  # one of job 5317993's eight


def api(method, path, key, body=None, timeout=60, accept="application/json"):
    req = urllib.request.Request(
        path if path.startswith("http") else API + path,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"OPEN-SANDBOX-API-KEY": key, "Content-Type": "application/json", "Accept": accept},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return None, f"{type(e).__name__}: {e}"


def log(case, msg):
    print(f"{time.strftime('%H:%M:%S')} [{case}] {msg}", flush=True)


def short(text, n=600):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " ..."


def run_case(name, image, cpu, mem_mib, disk_gib, key, tag, wait_s, results):
    body = {
        "image": {"uri": image},
        "entrypoint": ["tail", "-f", "/dev/null"],
        "resourceLimits": {"cpu": cpu, "memory": f"{mem_mib}Mi", "ephemeral-storage": f"{disk_gib}Gi"},
        "secureAccess": True,
        "timeout": wait_s + 600,  # lifetime; a backstop if this script dies
        "metadata": {"probe": tag, "probe-case": name},
    }
    log(name, f"create {image} cpu={cpu} memory={mem_mib}Mi ephemeral-storage={disk_gib}Gi")
    created = {}

    def create():
        # Breithorn's doc says create returns at once; the upstream server waits until ready
        created["resp"] = api("POST", "/sandboxes", key, body, timeout=wait_s + 120)

    t = threading.Thread(target=create, daemon=True)
    t0 = time.monotonic()
    t.start()
    sid, last, last_events, outcome = None, None, None, "timeout"
    while time.monotonic() - t0 < wait_s:
        if sid is None and "resp" in created:
            code, text = created["resp"]
            try:
                sid = json.loads(text).get("id")
            except ValueError:
                pass
            if sid is None:
                log(name, f"create -> HTTP {code}: {short(text)}")
                outcome = f"create HTTP {code}"
                break
        if sid is None:
            q = urllib.parse.urlencode({"metadata": urllib.parse.urlencode({"probe": tag, "probe-case": name})})
            code, text = api("GET", f"/sandboxes?{q}", key)
            try:
                items = json.loads(text).get("items") or []
                sid = items[0]["id"] if items else None
            except (ValueError, KeyError, IndexError, AttributeError):
                pass
        if sid is not None:
            code, text = api("GET", f"/sandboxes/{sid}", key)
            try:
                st = json.loads(text).get("status") or {}
                cur = (st.get("state"), st.get("reason"), st.get("message"))
            except ValueError:
                cur = (f"HTTP {code}", None, short(text, 200))
            if cur != last:
                log(name, f"+{time.monotonic() - t0:5.0f}s {sid} state={cur[0]} reason={cur[1]} message={cur[2]}")
                last = cur
            code, ev = api("GET", f"/sandboxes/{sid}/diagnostics/events?limit=20", key, accept="text/plain")
            ev = f"HTTP {code}: {ev}" if code != 200 else ev
            if ev != last_events:
                log(name, f"events: {short(ev, 1500)}")
                last_events = ev
            if cur[0] == "Running":
                outcome = f"Running after {time.monotonic() - t0:.0f}s"
                break
            if cur[0] in ("Failed", "Terminated"):
                outcome = f"{cur[0]} after {time.monotonic() - t0:.0f}s"
                break
        time.sleep(5)
    if "resp" in created and sid is not None and outcome == "timeout":
        code, text = created["resp"]
        log(name, f"create -> HTTP {code}: {short(text)}")
    if sid is not None:
        code, _ = api("DELETE", f"/sandboxes/{sid}", key)
        log(name, f"delete -> HTTP {code}")
    log(name, f"RESULT {outcome}")
    results[name] = outcome


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    scratch = os.environ.get("SCRATCH", "")
    p.add_argument("--task-toml", default=f"{scratch}/osb/swe_lego_v05/tasks/{SWE_DEFAULT_TASK}/task.toml")
    p.add_argument("--cases", default="small,swe-small,swe-full")
    a = p.parse_args()

    key_file = os.path.expanduser(os.environ.get("OPENSANDBOX_API_KEY_FILE", "~/.opensandbox_token"))
    key = open(key_file).read().strip()
    toml = open(a.task_toml).read()
    m = re.search(r'^docker_image = "([^"]+)"', toml, re.M)
    if not m:
        sys.exit(f"no docker_image in {a.task_toml}")
    swe_image = m.group(1)
    wait_s = int(os.environ.get("WAIT_S", "900"))
    tag = f"osb-pending-{time.strftime('%Y%m%d-%H%M%S')}"

    code, text = api("GET", API[: -len("/v1")] + "/version", key)  # outside /v1, no auth upstream
    print(f"API {API}; /version -> HTTP {code}: {short(text, 200)}; probe label {tag}; wait {wait_s}s", flush=True)
    cases = {
        "small": ("python:3.12-slim", "1", 1024, 2),
        "swe-small": (swe_image, "1", 1024, 2),
        "swe-full": (swe_image, "4", 4096, 10),
    }
    results = {}
    threads = [
        threading.Thread(target=run_case, args=(c, *cases[c], key, tag, wait_s, results))
        for c in a.cases.split(",")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("== summary")
    for c in a.cases.split(","):
        print(f"  {c:10s} {results.get(c)}")


API = os.environ.get("OPENSANDBOX_DOMAIN", "https://opensandbox-gvisor.breithorn.svc.cscs.ch").rstrip("/") + "/v1"

if __name__ == "__main__":
    main()
