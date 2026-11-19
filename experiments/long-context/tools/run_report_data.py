"""Per-step numbers of a STEM-mix run for its report page: timings, rewards, lengths, validations.
python3 run_report_data.py <job id> <total steps> <exp dir> <out.json>"""
import collections, glob, json, os, re, sys
SCRATCH = os.environ.get("SCRATCH") or "/scratch/" + os.environ.get("USER", "")
from datetime import datetime
job, total, exp, out = sys.argv[1], int(sys.argv[2]), sys.argv[3].rstrip("/"), sys.argv[4]
L = "%s/nemo-rl/slurm/%s-logs/ray-driver.log" % (SCRATCH, job)
keys = {"Total step time": "total", "policy_training": "train", "exposed_generation": "gen_wait", "weight_sync": "sync",
        "checkpointing": "ckpt", "idle/refit_bubble": "refit", "data_processing": "data"}
steps = collections.defaultdict(dict); vals = []; cur = None; curval = None; starts = []
for line in open(L, errors="replace"):
    m = re.search(r"= Step (\d+)/%d =" % total, line)
    if m: cur = int(m.group(1)); continue
    if "train start ts=" in line:
        m = re.search(r"train start ts=(\S+?)Z", line)
        if m: starts.append(datetime.strptime(m.group(1)[:19], "%Y-%m-%dT%H:%M:%S").timestamp())
        continue
    m = re.search(r"Starting validation at step (\d+)", line)
    if m: curval = {"step": int(m.group(1))}; vals.append(curval); continue
    if curval is not None:
        for k, pat in (("overall", r"• Accuracy: ([\d.]+)"), ("aime", r"accuracy_DAPOMathAIME2024: ([\d.]+)"), ("gpqa", r"accuracy_stem-rl-gpqa_diamond_x8: ([\d.]+)"),
                       ("chem", r"accuracy_chem_faithful-validation: ([\d.]+)"), ("len", r"Average response length: ([\d.]+)"), ("time", r"Total validation time: ([\d.]+)")):
            m = re.search(pat, line)
            if m and k not in curval: curval[k] = float(m.group(1))
    if cur is None: continue
    s = steps[cur]
    m = re.search(r"• ([A-Za-z_/ ]+): ([\d.]+)s", line)
    if m and m.group(1).strip() in keys: s.setdefault(keys[m.group(1).strip()], float(m.group(2)))
    for k, pat in (("reward", r"Avg Reward: ([\d.]+)"), ("age", r"Avg Trajectory Age: ([\d.]+)"), ("kl", r"Generation KL Error: ([\d.eE+-]+)"), ("loss", r"• Loss: (-?[\d.eE+-]+)")):
        m = re.search(pat, line)
        if m and k not in s: s[k] = float(m.group(1))
starts.sort(); ranks = 64
t0 = [starts[i] for i in range(0, len(starts), ranks)]
ds = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0, 0, 0.0]))
for p in glob.glob(exp + "/rollout_index/step*.tsv"):
    st = int(re.search(r"step(\d+)\.tsv", p).group(1))
    with open(p) as f:
        next(f)
        for line in f:
            _, _, _, name, _, rew, length = line.rstrip("\n").split("\t")
            for k in (name, "ALL"):
                d = ds[k][st]; d[0] += 1; d[1] += float(rew); d[2] += float(rew) >= 1.0; d[3] += float(length)
R = lambda v, n=4: None if v is None else round(v, n)
data = {"job": job, "steps": total,
        "timing": {k: [R(steps[i].get(k), 1) for i in range(1, total + 1)] for k in keys.values()},
        "reward": [R(steps[i].get("reward")) for i in range(1, total + 1)],
        "age": [R(steps[i].get("age"), 2) for i in range(1, total + 1)],
        "kl": [R(steps[i].get("kl"), 5) for i in range(1, total + 1)],
        "loss": [R(steps[i].get("loss"), 5) for i in range(1, total + 1)],
        "train_start_utc": [datetime.utcfromtimestamp(t).strftime("%H:%M:%S") for t in t0[:total]],
        "validation": vals,
        "datasets": {k: {"reward": [R(v[i][1] / v[i][0]) if v[i][0] else None for i in range(1, total + 1)],
                         "correct": [R(v[i][2] / float(v[i][0])) if v[i][0] else None for i in range(1, total + 1)],
                         "tokens": [int(v[i][3] / v[i][0]) if v[i][0] else None for i in range(1, total + 1)]} for k, v in ds.items()}}
json.dump(data, open(out, "w"), separators=(",", ":"))
print("steps with timing", sum(1 for x in data["timing"]["total"] if x is not None), "| validations", len(vals), "| datasets", sorted(data["datasets"]), "| step starts", len(t0))
for k in ("age", "kl", "loss"): print(k, [x for x in data[k] if x is not None][:3], "...", sum(1 for x in data[k] if x is not None))
