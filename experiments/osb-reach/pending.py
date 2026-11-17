#!/usr/bin/env python3
"""Why do OpenSandbox pods stay Pending? Ask the API instead of kubectl.

Job 5317993 (SWE-Lego, 4 CPU / 4 GiB / 10 GiB per pod): all 512 creates ended in
HTTP 504 KUBERNETES::POD_READY_TIMEOUT, "Last state: Pending" after 600 s. The server
deletes a sandbox when its ready wait fails, so its state has to be read while it waits.

Creates one sandbox per case, at the same time, then polls GET /v1/sandboxes/{id} and the
Kubernetes events (GET /v1/sandboxes/{id}/diagnostics/events) until it runs or the wait ends,
and deletes it. Every case sends the run's placeholder registry credentials (image_auth, as
run-s-sft.yaml) unless it says noauth. The cases tell the causes apart:
  small           python:3.12-slim, 1 CPU / 1024 MiB / 2 GiB (as the micro tasks, which run)
  swe-small       the SWE-Lego image, the micro tasks' resources            -> does the image start?
  swe-full        the SWE-Lego image, the task's 4 CPU / 4096 MiB / 10 GiB  -> exactly job 5317993's pods
  swe-req         as swe-full, requests 0.5 CPU / 512 MiB (Gym's upstream training config; ours drops it)
  swe-full-noauth as swe-full without image_auth                            -> do the placeholder creds hurt?
swe-small runs, swe-full stays Pending: scheduling/quota (swe-req running = the fix). Both stay Pending:
the image (pull); swe-full-noauth running = the credentials.

The task's image and its resources (swe-full, swe-req, swe-full-noauth) come from the task.toml's [environment]
section (the agent pod), not from [verifier.environment]: verified tasks with a separate verifier have both.
Registry credentials: RL_REGISTRY_USERNAME/RL_REGISTRY_PASSWORD if set, else the `name:password` line of
RL_REGISTRY_CRED_FILE (default ~/.rl_registry_cred), else anonymous/unused (the old placeholder).
Exit 0 when every case reached Running, 1 otherwise.

Standard library only; the API key and the registry password are read from files and never printed.
  python3 experiments/osb-reach/pending.py [--task-toml $SCRATCH/osb/swe_lego_v05/tasks/<task>/task.toml]
Env: OPENSANDBOX_API_KEY_FILE (default ~/.opensandbox_token), OPENSANDBOX_DOMAIN
(default https://opensandbox-gvisor.breithorn.svc.cscs.ch), WAIT_S (default 900), RL_REGISTRY_CRED_FILE.
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


def run_case(name, image, cpu, mem_mib, disk_gib, requests, auth, key, tag, wait_s, results):
    body = {
        "image": {"uri": image},
        "entrypoint": ["tail", "-f", "/dev/null"],
        "resourceLimits": {"cpu": cpu, "memory": f"{mem_mib}Mi", "ephemeral-storage": f"{disk_gib}Gi"},
        "secureAccess": True,
        "timeout": wait_s + 600,  # lifetime; a backstop if this script dies
        "metadata": {"probe": tag, "probe-case": name},
    }
    if requests:
        body["resourceRequests"] = {"cpu": requests[0], "memory": f"{requests[1]}Mi"}
    if auth:
        body["image"]["auth"] = {"username": REGISTRY_AUTH[0], "password": REGISTRY_AUTH[1]}
    log(name, f"create {image} cpu={cpu} memory={mem_mib}Mi ephemeral-storage={disk_gib}Gi "
              f"requests={requests} image_auth={'yes' if auth else 'no'}")
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
    p.add_argument("--cases", default="small,swe-small,swe-full,swe-req,swe-full-noauth")
    a = p.parse_args()

    key_file = os.path.expanduser(os.environ.get("OPENSANDBOX_API_KEY_FILE", "~/.opensandbox_token"))
    key = open(key_file).read().strip()
    env = environment_section(open(a.task_toml).read())
    if "docker_image" not in env:
        sys.exit(f"no docker_image in the [environment] section of {a.task_toml}")
    swe_image = env["docker_image"]
    # the task's own agent-pod resources (fallback: the SWE-Lego tasks' 4 CPU / 4096 MiB / 10 GiB)
    t_cpu = str(env.get("cpus", "4")).rstrip("0").rstrip(".") or "4"
    t_mem = int(float(env.get("memory_mb", 4096)))
    t_disk = max(1, -(-int(float(env.get("storage_mb", 10240))) // 1024))
    global REGISTRY_AUTH
    REGISTRY_AUTH, auth_src = registry_auth()
    print(f"task {a.task_toml}: image {swe_image}, cpu {t_cpu}, memory {t_mem}Mi, disk {t_disk}Gi; "
          f"registry credentials from {auth_src} (name {REGISTRY_AUTH[0]})", flush=True)
    wait_s = int(os.environ.get("WAIT_S", "900"))
    tag = f"osb-pending-{time.strftime('%Y%m%d-%H%M%S')}"

    code, text = api("GET", API[: -len("/v1")] + "/version", key)  # outside /v1, no auth upstream
    print(f"API {API}; /version -> HTTP {code}: {short(text, 200)}; probe label {tag}; wait {wait_s}s", flush=True)
    cases = {
        # image, cpu limit, memory MiB, disk GiB, (cpu, memory MiB) requests, image_auth
        "small": ("python:3.12-slim", "1", 1024, 2, None, True),
        "swe-small": (swe_image, "1", 1024, 2, None, True),
        "swe-full": (swe_image, t_cpu, t_mem, t_disk, None, True),
        "swe-req": (swe_image, t_cpu, t_mem, t_disk, ("0.5", 512), True),
        "swe-full-noauth": (swe_image, t_cpu, t_mem, t_disk, None, False),
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
        print(f"  {c:16s} {results.get(c)}")
    return 0 if all(str(results.get(c, "")).startswith("Running") for c in a.cases.split(",")) else 1


def environment_section(toml):
    """key = value pairs of the [environment] table only (standard library before tomllib; flat values)."""
    out, section = {}, None
    for line in toml.splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line.strip("[] ")
            continue
        m = re.match(r'^([A-Za-z_][\w-]*)\s*=\s*(.+?)\s*(#.*)?$', line)
        if section == "environment" and m:
            out[m.group(1)] = m.group(2).strip('"')
    return out


def registry_auth():
    """(name, password) and where they came from; the password is never printed."""
    if os.environ.get("RL_REGISTRY_USERNAME"):
        return (os.environ["RL_REGISTRY_USERNAME"], os.environ.get("RL_REGISTRY_PASSWORD", "")), "RL_REGISTRY_USERNAME/PASSWORD"
    path = os.path.expanduser(os.environ.get("RL_REGISTRY_CRED_FILE", "~/.rl_registry_cred"))
    try:
        line = open(path).readline().strip()
    except OSError:
        line = ""
    if ":" in line and line.split(":", 1)[0] and line.split(":", 1)[1]:
        name, pw = line.split(":", 1)
        return (name, pw), path
    return ("anonymous", "unused"), "the anonymous/unused placeholder"


API = os.environ.get("OPENSANDBOX_DOMAIN", "https://opensandbox-gvisor.breithorn.svc.cscs.ch").rstrip("/") + "/v1"

REGISTRY_AUTH = ("anonymous", "unused")

if __name__ == "__main__":
    sys.exit(main())
