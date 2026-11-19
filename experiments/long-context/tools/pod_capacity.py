"""How many OpenSandbox pods can we hold at once before the cluster queues or slows them?

Ramps the number of pods up in steps, with the same pod spec as the code_gen pool (Gym's opensandbox_pods.py:
python:3.12-slim, 1 CPU, 4 GiB, secureAccess). At each level it records how long the new pods took to answer, how
many failed and why, then runs the same small CPU task in every pod at once: if the pods are not really given a CPU
each, that task gets slower as the level rises. It stops ramping at the first level with create failures or slow
starts, and deletes every pod it made, by id, at the end (the cluster is shared: nothing else is touched).

    python3 pod_capacity.py --api-key-file FILE [--levels 64,128,192,256] [--out results.json]

The API key is read from the file into a request header and never printed. Pods only ever get their own endpoint
headers, as in the pool.
"""
import argparse
import json
import os
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--domain", default="https://opensandbox-gvisor.breithorn.svc.cscs.ch")
ap.add_argument("--api-key-file", required=True)
ap.add_argument("--levels", default="64,128,192,256")
ap.add_argument("--pod-seconds", type=int, default=900, help="pods expire by themselves after this long")
ap.add_argument("--ready-seconds", type=int, default=240, help="a pod that has not answered by then counts as failed")
ap.add_argument("--slow-start-seconds", type=float, default=90.0, help="stop ramping when the median start exceeds this")
ap.add_argument("--sustain-seconds", type=float, default=0.0,
                help="at each level, also run the CPU task in every pod back to back for this long, over kept-alive "
                     "connections, and report jobs per second: the load a run puts on the pods")
ap.add_argument("--out", default="")
args = ap.parse_args()

API = args.domain.rstrip("/") + "/v1"
KEY = {"OPEN-SANDBOX-API-KEY": open(args.api_key_file).read().strip()}
RUN = "podcap-" + uuid.uuid4().hex[:8]
EXECD_PORT = 44772
CPU_TASK = "python3 -c \"import time;t=time.time();s=sum(i*i for i in range(6000000));print('cpu',round(time.time()-t,2))\""
created = {}   # pod id -> (url, headers)
log = lambda *a: print(time.strftime("%T"), *a, flush=True)  # noqa: E731


def stdout(stream):
    out = []
    for line in stream.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if line.startswith("{"):
            event = json.loads(line)
            if event.get("type") == "stdout":
                out.append(event.get("text", ""))
    return "".join(out)


def run_in(pod, command, seconds=120):
    url, headers = pod
    r = requests.post(url + "/command", headers=headers, json={"command": command}, timeout=seconds)
    if r.status_code >= 300:
        raise RuntimeError("command: HTTP %d" % r.status_code)
    return stdout(r.text)


def create(_):
    t = time.time()
    try:
        r = requests.post(API + "/sandboxes", headers=KEY, timeout=args.ready_seconds, json={
            "image": {"uri": "python:3.12-slim"}, "timeout": args.pod_seconds, "secureAccess": True,
            "platform": {"os": "linux", "arch": "amd64"}, "resourceLimits": {"cpu": "1", "memory": "4Gi"},
            "entrypoint": ["tail", "-f", "/dev/null"],
            "metadata": {"nemo-gym.nvidia.com/user": os.environ.get("USER", "unknown"),
                         "nemo-gym.nvidia.com/workload": "pod-capacity-test", "nemo-gym.nvidia.com/run": RUN}})
        if r.status_code >= 300:
            return None, time.time() - t, "create: HTTP %d: %s" % (r.status_code, r.text[:200].replace("\n", " "))
        pod_id = r.json()["id"]
    except Exception as e:  # noqa: BLE001
        return None, time.time() - t, "create: %r" % (e,)
    created[pod_id] = None
    deadline = t + args.ready_seconds
    error = "never answered"
    while time.time() < deadline:
        try:
            r = requests.get("%s/sandboxes/%s/endpoints/%d" % (API, pod_id, EXECD_PORT), headers=KEY, timeout=30)
            if r.status_code < 300:
                e = r.json()
                pod = ("https://" + e["endpoint"], e.get("headers") or {})
                if "ok" in run_in(pod, "echo ok", 30):
                    created[pod_id] = pod
                    return pod_id, time.time() - t, None
            else:
                error = "endpoint: HTTP %d" % r.status_code
        except Exception as e:  # noqa: BLE001
            error = repr(e)[:160]
        time.sleep(2)
    return pod_id, time.time() - t, error


def delete(pod_id):
    try:
        return requests.delete("%s/sandboxes/%s" % (API, pod_id), headers=KEY, timeout=60).status_code
    except Exception as e:  # noqa: BLE001
        return repr(e)[:80]


def cpu_task(pod):
    t = time.time()
    try:
        out = run_in(pod, CPU_TASK, 180)
        inside = float(out.split("cpu")[1]) if "cpu" in out else None
        return time.time() - t, inside
    except Exception:  # noqa: BLE001
        return None, None


def sustain(pod):
    """Back-to-back CPU tasks in one pod over one kept-alive connection; wall time of each."""
    url, headers = pod
    session = requests.Session()
    walls, stop = [], time.time() + args.sustain_seconds
    while time.time() < stop:
        t = time.time()
        try:
            r = session.post(url + "/command", headers=headers, json={"command": CPU_TASK}, timeout=120)
            if r.status_code < 300 and "cpu" in stdout(r.text):
                walls.append(time.time() - t)
        except Exception:  # noqa: BLE001
            pass
    return walls


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


results = []
try:
    for level in (int(x) for x in args.levels.split(",")):
        have = sum(1 for p in created.values() if p)
        need = level - have
        log("level %d: creating %d pods (run label %s)" % (level, need, RUN))
        t0 = time.time()
        with ThreadPoolExecutor(need) as pool:
            made = list(pool.map(create, range(need)))
        ok = [m for m in made if m[2] is None]
        bad = [m for m in made if m[2] is not None]
        starts = [m[1] for m in ok]
        ready = [p for p in created.values() if p]
        with ThreadPoolExecutor(len(ready)) as pool:
            echo = []
            for p in pool.map(lambda pod: (time.time(), run_in(pod, "echo ok", 60), time.time()), ready):
                echo.append(p[2] - p[0])
            cpu = [c for c in pool.map(cpu_task, ready)]
        wall = [c[0] for c in cpu if c[0] is not None]
        inside = [c[1] for c in cpu if c[1] is not None]
        row = {"level": level, "asked": need, "started": len(ok), "failed": len(bad), "ramp_seconds": round(time.time() - t0, 1),
               "start_median_s": round(statistics.median(starts), 1) if starts else None,
               "start_p95_s": round(pct(starts, 0.95), 1) if starts else None,
               "pods_up": len(ready), "echo_median_s": round(statistics.median(echo), 2), "echo_p95_s": round(pct(echo, 0.95), 2),
               "cpu_task_inside_median_s": round(statistics.median(inside), 2) if inside else None,
               "cpu_task_inside_p95_s": round(pct(inside, 0.95), 2) if inside else None,
               "cpu_task_wall_median_s": round(statistics.median(wall), 2) if wall else None,
               "cpu_task_failed": len(cpu) - len(wall),
               "errors": sorted({m[2][:120] for m in bad})[:4]}
        if args.sustain_seconds > 0:
            t1 = time.time()
            with ThreadPoolExecutor(len(ready)) as pool:
                per_pod = list(pool.map(sustain, ready))
            walls = [w for ws in per_pod for w in ws]
            row.update(sustain_jobs=len(walls), sustain_jobs_per_s=round(len(walls) / (time.time() - t1), 1),
                       sustain_job_wall_median_s=round(statistics.median(walls), 2) if walls else None,
                       sustain_job_wall_p95_s=round(pct(walls, 0.95), 2) if walls else None,
                       sustain_idle_pods=sum(1 for ws in per_pod if not ws))
        results.append(row)
        log(json.dumps(row))
        if bad and len(bad) > max(2, need // 20):
            log("stopping: %d of %d pods did not start at level %d" % (len(bad), need, level))
            break
        if starts and statistics.median(starts) > args.slow_start_seconds:
            log("stopping: median start %.0f s at level %d" % (statistics.median(starts), level))
            break
finally:
    ids = list(created)
    log("deleting the %d pods this test made" % len(ids))
    if ids:
        with ThreadPoolExecutor(min(64, len(ids))) as pool:
            codes = list(pool.map(delete, ids))
        log("delete results:", dict((str(c), codes.count(c)) for c in set(codes)))
    if args.out:
        json.dump({"run": RUN, "levels": results}, open(args.out, "w"), indent=1)
print("\n%6s %6s %7s %7s %10s %9s %9s %11s %11s" % ("level", "asked", "started", "failed", "start med", "start p95", "echo med", "cpu inside", "cpu in p95"))
for r in results:
    print("%6d %6d %7d %7d %10s %9s %9s %11s %11s" % (r["level"], r["asked"], r["started"], r["failed"], r["start_median_s"],
                                                       r["start_p95_s"], r["echo_median_s"], r["cpu_task_inside_median_s"],
                                                       r["cpu_task_inside_p95_s"]))
if args.sustain_seconds > 0:
    print("\nsustained load: every pod runs the CPU task back to back")
    print("%6s %8s %10s %12s %12s" % ("level", "jobs", "jobs/s", "job wall med", "job wall p95"))
    for r in results:
        print("%6d %8s %10s %12s %12s" % (r["level"], r.get("sustain_jobs"), r.get("sustain_jobs_per_s"),
                                          r.get("sustain_job_wall_median_s"), r.get("sustain_job_wall_p95_s")))
