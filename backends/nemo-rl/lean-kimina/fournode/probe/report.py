#!/usr/bin/env python3
"""Report of one probe job (probe.sh): fit, resume and speed of the 4-node colocated recipe, with a verdict per part.

  python3 report.py <OUT> <job> [--run RUN] [--ranks 16] [--first-steps 3] [--steps 5]

OUT is the probe's output root ($PROBE/nemo-rl). Reads, all under OUT:
  slurm/<job>-logs/         ray-driver.log (split into phase A and B at probe_driver.sh's marks), ray-cluster-*.log,
                            gpumem-<node>.csv, hostmem-<node>.tsv (probe_watch.sh), numa-hbm-<node>.log
  logs/<RUN>/               probe-phases.tsv, TensorBoard events (timing/train/* per step)
  results/<RUN>/            step_<N>/policy/weights/optimizer-scratch/ (linked moments), scratch-adam-live/ with
                            ScratchAdamW's per-step stats lines (rank<r>/hdo<n>.stats.jsonl)
Verdicts: GO, CHECK (works, but look: little headroom, missing data) and NO-GO (does not work as is).
Python 3.6+ (Clariden's login node), standard library only.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import tfevents  # noqa: E402

GIB = 1024 ** 3
OOM_GPU = re.compile(r"CUDA out of memory|OutOfMemoryError|cudaErrorMemoryAllocation|CUDA error: out of memory")
OOM_HOST = re.compile(r"running low on memory|oom-kill|Out of memory: Killed process|Cannot allocate memory")
SCRATCH_ERR = re.compile(r"ScratchAdamW step failed|restore_scratch_state|reset_scratch_state|"
                         r"saved without ScratchAdamW moments")
MOMENTS = re.compile(r"ScratchAdamW moments: (\w+)")


def read_lines(path):
    try:
        with open(path, errors="replace") as f:
            return f.read().splitlines()
    except OSError:
        return []


def find_run(out, job):
    for path in sorted(glob.glob(os.path.join(out, "slurm", "%s-*.out" % job))):
        name = os.path.basename(path)[len(str(job)) + 1:-len(".out")]
        if not name.endswith(".preflight"):
            return name
    return None


class Report:
    def __init__(self):
        self.lines = []
        self.verdicts = []

    def say(self, text=""):
        self.lines.append(text)

    def verdict(self, part, value, why):
        self.verdicts.append((part, value, why))
        self.say("  -> %s: %s" % (value, why))

    def table(self, header, rows):
        widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
        fmt = "  " + "  ".join("%%-%ds" % w for w in widths)
        self.say(fmt % tuple(header))
        for r in rows:
            self.say(fmt % tuple(str(x) for x in r))

    def text(self):
        order = {"NO-GO": 0, "CHECK": 1, "GO": 2}
        overall = min((v for _, v, _ in self.verdicts), key=order.get, default="CHECK")
        head = ["Probe verdict: %s" % overall] + ["  %-10s %-6s %s" % v for v in self.verdicts] + [""]
        return "\n".join(head + self.lines) + "\n"


def phases(log_dir):
    marks = []
    for line in read_lines(os.path.join(log_dir, "probe-phases.tsv")):
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].isdigit():
            marks.append((int(parts[0]), parts[1], parts[2]))
    return marks


def split_driver_log(lines):
    """(phase A lines, phase B lines) at probe_driver.sh's 'phase B start' mark; all in A without it."""
    for i, line in enumerate(lines):
        if "[probe_driver]" in line and "phase B start" in line:
            return lines[:i], lines[i:]
    return lines, []


def scalars(log_dir):
    series = defaultdict(dict)
    files = sorted(glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True), key=os.path.getmtime)
    for path in files:                      # later files win: phase B's driver writes a new experiment folder
        for tag, step, value in tfevents.scalars(path):
            series[tag][step] = value
    return series


def gpu_part(rep, job_logs, mark_list, oom_lines):
    rep.say("GPU memory (nvidia-smi, every process on the GPU)")
    rows, worst = [], None
    for path in sorted(glob.glob(os.path.join(job_logs, "gpumem-*.csv"))):
        node = os.path.basename(path)[len("gpumem-"):-len(".csv")]
        peak = {}
        for line in read_lines(path):
            f = [x.strip() for x in line.split(",")]
            if len(f) < 4 or not f[2].isdigit():
                continue
            gpu, used, total = int(f[1]), int(f[2]) / 1024.0, int(f[3]) / 1024.0
            if gpu not in peak or used > peak[gpu][0]:
                peak[gpu] = (used, total, f[0])
        for gpu in sorted(peak):
            used, total, ts = peak[gpu]
            rows.append((node, gpu, "%.1f" % used, "%.1f" % total, "%.1f" % (total - used), when(ts, mark_list)))
            if worst is None or total - used < worst:
                worst = total - used
    rep.table(["node", "gpu", "peak GiB", "total GiB", "headroom", "phase at peak"], rows)
    if oom_lines:
        rep.say("  GPU out-of-memory lines: %d, first: %s" % (len(oom_lines), oom_lines[0][:200]))
        rep.verdict("GPU", "NO-GO", "out of GPU memory (%d lines)" % len(oom_lines))
    elif worst is None:
        rep.verdict("GPU", "CHECK", "no gpumem-*.csv data")
    elif worst < 2.0:
        rep.verdict("GPU", "CHECK", "fits, but the fullest GPU had %.1f GiB left" % worst)
    else:
        rep.verdict("GPU", "GO", "fits; the fullest GPU had %.1f GiB left" % worst)
    rep.say()


def when(ts, mark_list):
    """The probe phase a nvidia-smi timestamp (node-local time) falls into."""
    try:
        t = datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp()
    except ValueError:
        return "?"
    phase = "before A"
    for epoch, name, event in mark_list:
        if epoch <= t:
            phase = name if event.startswith("start") else "after " + name
    return phase


def host_part(rep, job_logs, host_min_gb, oom_lines):
    rep.say("Host memory (MemAvailable, and the free memory of each CPU NUMA node)")
    rows, worst = [], None
    for path in sorted(glob.glob(os.path.join(job_logs, "hostmem-*.tsv"))):
        node = os.path.basename(path)[len("hostmem-"):-len(".tsv")]
        low, total, numa_low = None, None, {}
        for line in read_lines(path):
            kv = dict(x.split("=", 1) for x in line.split("\t")[1:] if "=" in x)
            try:
                avail, total = float(kv["MemAvailable"]), float(kv["MemTotal"])
            except (KeyError, ValueError):
                continue
            low = avail if low is None else min(low, avail)
            for k, v in kv.items():
                if k.startswith("cpufree"):
                    numa_low[k[len("cpufree"):]] = min(numa_low.get(k[len("cpufree"):], 1e9), float(v))
        if low is None:
            continue
        rows.append((node, "%.0f" % total, "%.0f" % (total - low), "%.0f" % low,
                     " ".join("%s:%.0f" % (k, numa_low[k]) for k in sorted(numa_low))))
        worst = low if worst is None else min(worst, low)
    rep.table(["node", "total GiB", "peak used", "min available", "min free per CPU NUMA node"], rows)
    spills = hbm_spills(job_logs)
    if spills:
        rep.say("  host memory on GPU HBM NUMA nodes (numa_watch), max GiB: " +
                ", ".join("%s node%s %.1f" % s for s in spills))
    if oom_lines:
        rep.say("  host out-of-memory lines: %d, first: %s" % (len(oom_lines), oom_lines[0][:200]))
        rep.verdict("host", "NO-GO", "out of host memory (%d lines)" % len(oom_lines))
    elif worst is None:
        rep.verdict("host", "CHECK", "no hostmem-*.tsv data")
    elif worst < host_min_gb or spills:
        rep.verdict("host", "CHECK", "lowest MemAvailable %.0f GiB (want >= %d)%s" % (
            worst, host_min_gb, ", host memory spilled onto HBM" if spills else ""))
    else:
        rep.verdict("host", "GO", "lowest MemAvailable %.0f GiB" % worst)
    rep.say()


def hbm_spills(job_logs, limit_gb=2.0):
    worst = {}
    pat = re.compile(r"node(\d+) used=[\d.]+G file=([\d.]+)G anon=([\d.]+)G shmem=([\d.]+)G")
    for path in glob.glob(os.path.join(job_logs, "numa-hbm-*.log")):
        node = os.path.basename(path)[len("numa-hbm-"):-len(".log")]
        for line in read_lines(path):
            for m in pat.finditer(line):
                gb = sum(float(x) for x in m.groups()[1:])
                key = (node, m.group(1))
                worst[key] = max(worst.get(key, 0.0), gb)
    return sorted((n, h, gb) for (n, h), gb in worst.items() if gb > limit_gb)


def scratch_part(rep, ckpt_dir, ranks):
    rep.say("ScratchAdamW (m, v on scratch): one line per optimizer step and rank")
    per_step = defaultdict(list)
    for path in glob.glob(os.path.join(ckpt_dir, "scratch-adam-live", "rank*", "hdo*.stats.jsonl")):
        name = os.path.relpath(path, os.path.join(ckpt_dir, "scratch-adam-live"))
        for line in read_lines(path):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            per_step[e.get("step")].append(dict(e, name=name))
    rows = []
    for step in sorted(s for s in per_step if s is not None):
        es = per_step[step]
        by_host = defaultdict(lambda: [0.0, 0.0])
        for e in es:
            by_host[e["host"]][0] += e["io_GBps"] * e["seconds"]
            by_host[e["host"]][1] = max(by_host[e["host"]][1], e["seconds"])
        node_rate = min(b / s for b, s in by_host.values() if s > 0)
        rows.append((step, len(es), "%.0f" % max(e["seconds"] for e in es),
                     "%.0f" % statistics.median(e["seconds"] for e in es),
                     "%.0f" % max(e["compute_seconds"] for e in es),
                     "%.0f" % max(b for b, _ in by_host.values()), "%.1f" % node_rate))
    rep.table(["step", "optimizers", "max s", "median s", "max compute s", "max GB per node", "min node GB/s"], rows)
    if not rows:
        rep.verdict("scratch", "CHECK", "no ScratchAdamW stats lines")
    elif any(int(r[1]) < ranks for r in rows):
        rep.verdict("scratch", "CHECK", "some steps have fewer than %d optimizers reporting" % ranks)
    else:
        rep.verdict("scratch", "GO", "every step on every rank; slowest step %s s" % max(int(r[2]) for r in rows))
    rep.say()


def resume_part(rep, a_lines, b_lines, ranks, first, steps, series, marks):
    rep.say("Checkpoint and resume")
    a = [m.group(1) for m in map(MOMENTS.search, a_lines) if m]
    b = [m.group(1) for m in map(MOMENTS.search, b_lines) if m]
    errors = [x for x in a_lines + b_lines if SCRATCH_ERR.search(x) and "[probe_driver]" not in x]
    done = sorted(series.get("timing/train/total_step_time", {}))
    rep.say("  phase A moments: %s; phase B moments: %s" % (count(a), count(b)))
    rep.say("  steps logged: %s" % (done or "none"))
    for epoch, phase, event in marks:
        rep.say("  %s phase %s %s" % (datetime.fromtimestamp(epoch).strftime("%H:%M:%S"), phase, event))
    ended = {(p, e.split()[1]) for _, p, e in marks if e.startswith("end")}
    want_steps = list(range(1, steps + 1))
    if errors:
        rep.say("  ScratchAdamW errors: %d, first: %s" % (len(errors), errors[0][:200]))
        rep.verdict("resume", "NO-GO", "ScratchAdamW refused or failed (%d lines)" % len(errors))
    elif b.count("restored") == ranks and a.count("reset") == ranks and done == want_steps and ("B", "rc=0") in ended:
        rep.verdict("resume", "GO", "fresh start on %d ranks, all %d restored from step_%d, steps %d-%d ran" % (
            ranks, ranks, first, first + 1, steps))
    elif ("B", "rc=0") in ended or done:
        rep.verdict("resume", "CHECK", "the run went on, but not as expected (moments or steps above)")
    else:
        rep.verdict("resume", "NO-GO", "phase B did not complete")
    rep.say()


def count(xs):
    c = defaultdict(int)
    for x in xs:
        c[x] += 1
    return ", ".join("%d %s" % (n, k) for k, n in sorted(c.items())) or "none"


def checkpoint_part(rep, ckpt_dir, ranks, first, series):
    rows = []
    for d in sorted(glob.glob(os.path.join(ckpt_dir, "step_*")), key=lambda p: int(p.rsplit("_", 1)[1])):
        scratch = os.path.join(d, "policy", "weights", "optimizer-scratch")
        manifests = glob.glob(os.path.join(scratch, "rank*", "hdo*", "manifest.json"))
        other = linked = 0
        for root, _, files in os.walk(d):
            for f in files:
                st = os.stat(os.path.join(root, f))
                if root.startswith(scratch):
                    linked += st.st_size
                else:
                    other += st.st_size
        step = int(d.rsplit("_", 1)[1])
        secs = series.get("timing/train/checkpointing", {}).get(step)
        rows.append((os.path.basename(d), len(manifests), "%.0f" % (other / 1e9), "%.0f" % (linked / 1e9),
                     "-" if secs is None else "%.0f" % secs))
    rep.say("Checkpoints kept")
    rep.table(["checkpoint", "moment manifests", "GB written", "GB linked moments", "save s"], rows)
    mine = [r for r in rows if r[0] == "step_%d" % first]
    if not mine:
        rep.verdict("checkpoint", "NO-GO", "no step_%d" % first)
    elif mine[0][1] < ranks:
        rep.verdict("checkpoint", "NO-GO", "step_%d has %d moment manifests, want >= %d" % (first, mine[0][1], ranks))
    else:
        rep.verdict("checkpoint", "GO", "step_%d holds the moments of all ranks" % first)
    rep.say()


def timing_part(rep, series, steps):
    tags = sorted(t for t in series if t.startswith("timing/train/") and t.count("/") == 2)
    rep.say("Phase times per step (s, timing/train/*)")
    cols = list(range(1, steps + 1))
    rows = []
    for t in tags:
        vals = series[t]
        if not any(s in vals for s in cols):
            continue
        rows.append([t[len("timing/train/"):]] + ["-" if s not in vals else "%.0f" % vals[s] for s in cols])
    rows.sort(key=lambda r: r[0] != "total_step_time")
    rep.table(["phase"] + ["step %d" % s for s in cols], rows)
    rep.say()


def mismatch_part(rep, series, steps):
    """Train/inference mismatch per step (the quantities router replay should lower), whichever of them are logged."""
    tags = sorted(t for t in series if t.startswith("train/") and
                  re.search(r"mult_prob_error|gen_kl|logprob_error|masked|r3", t))
    rep.say("Train/inference mismatch per step (train/*; compare a probe with R3=1 against one without)")
    cols = list(range(1, steps + 1))
    rows = [[t[len("train/"):]] + ["-" if s not in series[t] else "%.4g" % series[t][s] for s in cols] for t in tags]
    rep.table(["metric"] + ["step %d" % s for s in cols], rows)
    rep.say()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("job")
    ap.add_argument("--run")
    ap.add_argument("--ranks", type=int, default=16, help="trainer ranks (GPUs), default 16")
    ap.add_argument("--first-steps", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--host-min-gb", type=int, default=16, help="CHECK below this MemAvailable (GiB), default 16")
    args = ap.parse_args(argv)
    run = args.run or find_run(args.out, args.job)
    if not run:
        sys.exit("no %s/slurm/%s-<run>.out; give --run" % (args.out, args.job))
    job_logs = os.path.join(args.out, "slurm", "%s-logs" % args.job)
    log_dir = os.path.join(args.out, "logs", run)
    ckpt_dir = os.path.join(args.out, "results", run)
    driver = read_lines(os.path.join(job_logs, "ray-driver.log"))
    others = [x for p in sorted(glob.glob(os.path.join(job_logs, "ray-cluster-*.log"))) for x in read_lines(p)]
    gpu_oom = [x for x in driver + others if OOM_GPU.search(x)]
    host_oom = [x for x in driver + others if OOM_HOST.search(x)]
    a_lines, b_lines = split_driver_log(driver)
    marks = phases(log_dir)
    series = scalars(log_dir)

    rep = Report()
    rep.say("Probe job %s, run %s (%s)" % (args.job, run, args.out))
    rep.say()
    gpu_part(rep, job_logs, marks, gpu_oom)
    host_part(rep, job_logs, args.host_min_gb, host_oom)
    resume_part(rep, a_lines, b_lines, args.ranks, args.first_steps, args.steps, series, marks)
    checkpoint_part(rep, ckpt_dir, args.ranks, args.first_steps, series)
    scratch_part(rep, ckpt_dir, args.ranks)
    timing_part(rep, series, args.steps)
    mismatch_part(rep, series, args.steps)
    sys.stdout.write(rep.text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
