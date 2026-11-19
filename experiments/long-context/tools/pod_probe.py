"""Health probe of the OpenSandbox service, for when a run's pods fail with "the command did not finish".

Makes a few pods the way a run's code_gen server does and runs commands of growing length in each
("sleep N; echo marker"). A command whose output lacks the marker was cut off; the probe prints after how long
and what the stream ended with. Deletes its pods at the end.

  python3 pod_probe.py --api-key-file ~/.opensandbox_token [--pods 3] [--sleeps 1,5,15,30,60,120]
"""
import argparse
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--domain", default="https://opensandbox-gvisor.breithorn.svc.cscs.ch")
ap.add_argument("--api-key-file", required=True)
ap.add_argument("--pods", type=int, default=3)
ap.add_argument("--sleeps", default="1,5,15,30,60,120")
ap.add_argument("--ready-seconds", type=int, default=240)
args = ap.parse_args()

API = args.domain.rstrip("/") + "/v1"
KEY = {"OPEN-SANDBOX-API-KEY": open(args.api_key_file).read().strip()}
RUN = "podprobe-" + uuid.uuid4().hex[:8]
EXECD_PORT = 44772
MARK = "##probe-done##"
log = lambda *a: print(time.strftime("%T"), *a, flush=True)  # noqa: E731


def events(stream):
    """(stdout text, event types in order) of a command's event stream."""
    out, kinds = [], []
    for line in stream.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if line.startswith("{"):
            try:
                event = json.loads(line)
            except ValueError:
                kinds.append("unparsed")
                continue
            kinds.append(str(event.get("type")))
            if event.get("type") == "stdout":
                out.append(event.get("text", ""))
    return "".join(out), kinds


def command(pod, text, seconds):
    url, headers = pod
    t = time.time()
    try:
        r = requests.post(url + "/command", headers=headers, json={"command": text}, timeout=seconds)
    except Exception as e:  # noqa: BLE001
        return time.time() - t, None, "", [repr(e)[:160]], ""
    out, kinds = events(r.text)
    return time.time() - t, r.status_code, out, kinds, r.text[-300:].replace("\n", " | ")


def create(i):
    t = time.time()
    r = requests.post(API + "/sandboxes", headers=KEY, timeout=args.ready_seconds, json={
        "image": {"uri": "python:3.12-slim"}, "timeout": 900, "secureAccess": True,
        "platform": {"os": "linux", "arch": "amd64"}, "resourceLimits": {"cpu": "1", "memory": "4Gi"},
        "entrypoint": ["tail", "-f", "/dev/null"],
        "metadata": {"nemo-gym.nvidia.com/user": os.environ.get("USER", "unknown"),
                     "nemo-gym.nvidia.com/workload": "pod-probe", "nemo-gym.nvidia.com/run": RUN}})
    if r.status_code >= 300:
        return i, None, None, "create: HTTP %d: %s" % (r.status_code, r.text[:200].replace("\n", " "))
    pod_id = r.json()["id"]
    deadline = t + args.ready_seconds
    error = "never answered"
    while time.time() < deadline:
        try:
            r = requests.get("%s/sandboxes/%s/endpoints/%d" % (API, pod_id, EXECD_PORT), headers=KEY, timeout=30)
            if r.status_code < 300:
                e = r.json()
                pod = ("https://" + e["endpoint"], e.get("headers") or {})
                _, status, out, _, _ = command(pod, "echo ok", 30)
                if status is not None and status < 300 and "ok" in out:
                    log("pod %d (%s) answers after %.0f s" % (i, pod_id[:8], time.time() - t))
                    return i, pod_id, pod, None
                error = "echo: HTTP %s, output %r" % (status, out[:60])
            else:
                error = "endpoint: HTTP %d" % r.status_code
        except Exception as e:  # noqa: BLE001
            error = repr(e)[:160]
        time.sleep(2)
    return i, pod_id, None, error


def probe(item):
    i, pod_id, pod, error = item
    if pod is None:
        log("pod %d: NOT USABLE: %s" % (i, error))
        return
    for seconds in [int(s) for s in args.sleeps.split(",")]:
        wall, status, out, kinds, tail = command(pod, "sleep %d; echo '%s'" % (seconds, MARK), seconds + 90)
        ok = out.strip().endswith(MARK)
        log("pod %d: sleep %3d s -> %s after %.1f s (HTTP %s, events %s)%s" % (
            i, seconds, "finished" if ok else "CUT OFF", wall, status, ",".join(kinds[-4:]) or "none",
            "" if ok else "  stream ends: " + tail))
    # What a submission does besides waiting: Python start-up and a second of CPU.
    wall, status, out, kinds, tail = command(
        pod, "python3 -c \"s=sum(i*i for i in range(6000000)); print('cpu ok')\"; echo '%s'" % MARK, 120)
    log("pod %d: python task -> %s after %.1f s (HTTP %s)" % (
        i, "finished" if out.strip().endswith(MARK) and "cpu ok" in out else "FAILED: " + tail, wall, status))


log("probe %s: %d pods at %s" % (RUN, args.pods, args.domain))
with ThreadPoolExecutor(args.pods) as pool:
    made = list(pool.map(create, range(args.pods)))
    list(pool.map(probe, made))
    for i, pod_id, _, _ in made:
        if pod_id:
            try:
                code = requests.delete("%s/sandboxes/%s" % (API, pod_id), headers=KEY, timeout=60).status_code
            except Exception as e:  # noqa: BLE001
                code = repr(e)[:80]
            log("pod %d deleted: %s" % (i, code))
log("probe done")
