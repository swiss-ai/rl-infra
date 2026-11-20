"""Per training step of an async GRPO run: vLLM worker busy and idle time (from the inflight-batch timeline the driver
prints), peak concurrent sequences, tokens per sample, rows trained, answers cut at the limit.
python3 gen_stats.py <job id> <total steps> <out.json>"""
import json, os, re, sys
SCRATCH = os.environ.get("SCRATCH") or "/scratch/" + os.environ.get("USER", "")
job, total, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
L = "%s/nemo-rl/slurm/%s-logs/ray-driver.log" % (SCRATCH, job)
res = {}; cur = None; inflight = False
pat_w = re.compile(r"Generation Worker\s+\d+: .*\(Active: ([\d.]+) s, Idle: ([\d.]+) s\)")
for line in open(L, errors="replace"):
    m = re.search(r"= Step (\d+)/%d =" % total, line)
    if m:
        cur = int(m.group(1)); res.setdefault(cur, {"active": [], "idle": []}); continue
    if cur is None: continue
    s = res[cur]
    m = pat_w.search(line)
    if m: s["active"].append(float(m.group(1))); s["idle"].append(float(m.group(2))); continue
    if "Inflight Batch Sizes" in line: inflight = True; continue
    if inflight and "Max value:" in line:
        s.setdefault("max_inflight", int(float(line.split("Max value:")[1]))); inflight = False; continue
    for k, p in (("trained_rows", r"Training (\d+) of \d+ rows"), ("truncated", r"(\d+)/\d+ samples truncated"),
                 ("tokens_per_sample", r"Mean Total Tokens per Sample: ([\d.]+)"), ("imbalance", r"Average Token Imbalance: ([\d.]+)%")):
        m = re.search(p, line)
        if m and k not in s: s[k] = float(m.group(1))
steps = []
for i in range(1, total + 1):
    s = res.get(i, {"active": [], "idle": []}); a, d = s["active"], s["idle"]
    win = (a[0] + d[0]) if a else None
    steps.append({"step": i, "workers": len(a), "window": win, "active_mean": round(sum(a) / len(a), 1) if a else None,
                  "active_min": min(a) if a else None, "active_max": max(a) if a else None, "busy": round(sum(a) / len(a) / win, 4) if a and win else None,
                  "max_inflight": s.get("max_inflight"), "trained_rows": s.get("trained_rows"), "truncated": s.get("truncated"),
                  "tokens_per_sample": s.get("tokens_per_sample"), "imbalance": s.get("imbalance")})
json.dump(steps, open(out, "w"), separators=(",", ":"))
ok = [x for x in steps if x["busy"] is not None]
print("steps with worker data", len(ok), "workers", ok[0]["workers"])
for x in steps[:3] + steps[58:61] + steps[-3:]: print(x)
b = sorted(x["busy"] for x in ok); print("busy share median %.3f p10 %.3f p90 %.3f" % (b[len(b)//2], b[len(b)//10], b[9*len(b)//10]))
w = sorted(x["window"] for x in ok); print("window median", w[len(w)//2])
t = [x["trained_rows"] for x in steps if x["trained_rows"]]; print("trained rows mean %.0f first %s last %s" % (sum(t)/len(t), t[:3], t[-3:]))
