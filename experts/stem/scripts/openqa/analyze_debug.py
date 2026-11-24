"""Summarises a judge debug run (hangang-dm, Python 3.6): step times, llm_judge call latency, judge pool load,
per-task training reward. Usage: python3 analyze_debug.py <job id> <run name>"""
import glob, json, re, sys
sys.path.insert(0, "/scratch/hanint06/rl-analysis")
import tbs_lib
J, RUN = sys.argv[1], sys.argv[2]
L = "/scratch/hanint06/nemo-rl/slurm/%s-logs/ray-driver.log" % J
txt = open(L, errors="replace").read()
steps = [float(x) for x in re.findall(r"Total step time: ([0-9.]+)s", txt)]
print("steps", len(steps), "step times", [round(s) for s in steps])
jl = re.findall(r"\[llm_judge\] calls=(\d+) failures=(\d+) (.*?)in_flight=(\d+) latency_s mean=([0-9.]+) p50=([0-9.]+) p95=([0-9.]+) t=(\d+)", txt)
print("llm_judge lines", len(jl))
for x in jl[::max(1, len(jl) // 12)] + jl[-1:]:
    print("  calls %s failures %s %s in_flight %s latency mean %s p50 %s p95 %s" % x[:7])
fails = re.findall(r"\[llm_judge\] call failed: (.*)", txt)
print("failed-call lines", len(fails), fails[:3])
st = glob.glob("/scratch/hanint06/nemo-rl/logs/%s/judge_stats.jsonl" % RUN) + glob.glob("/scratch/hanint06/nemo-rl/logs/%s/exp_*/judge_stats.jsonl" % RUN)
if st:
    rows = [json.loads(l) for l in open(st[0])]
    run_ = [r.get("num_requests_running", 0) for r in rows]; wait = [r.get("num_requests_waiting", 0) for r in rows]
    kv = [r.get("kv_cache_usage_perc", 0) for r in rows]
    busy = sum(1 for r in run_ if r and r > 0) / max(1, len(run_))
    print("judge stats: samples %d, share of samples with requests running %.2f, max running %s, max waiting %s, max kv %.2f" % (len(rows), busy, max(run_), max(wait), max(kv)))
files = glob.glob("/scratch/hanint06/nemo-rl/logs/%s/exp_*/tensorboard/**/events*" % RUN, recursive=True)
s = {}
for f in files:
    for k, v in tbs_lib.scalars(f, lambda t: t.startswith("train/reward_") or "mask_sample" in t or "judge" in t or t in ("train/truncation_rate", "timing/train/total_step_time", "timing/train/policy_training", "timing/train/exposed_generation", "timing/train/idle/buffer_starvation") ).items():
        s.setdefault(k, {}).update(dict(v) if not isinstance(v, dict) else v)
for k in sorted(s):
    print(k, {st: round(val, 3) for st, val in sorted(s[k].items())})
