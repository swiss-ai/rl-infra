"""Statistics of the rollouts of a STEM-mix run, per dataset: lengths, rewards, masks, turns, prompt coverage.
python3 rollout_stats.py <exp dir> <out.json>   (reads rollout_index/step*.tsv and train_data_step*.jsonl)"""
import collections, glob, json, re, sys
exp, out = sys.argv[1].rstrip("/"), sys.argv[2]
LIMIT = 32768
S = collections.defaultdict(lambda: {"n": 0, "len": [], "r0": 0, "r1": 0, "rp": 0, "rsum": 0.0, "rows": set(), "masked": 0, "msgs": collections.Counter(),
                                     "chars": 0, "groups": 0, "g_all": 0, "g_none": 0, "g_some": 0})
names = {}
for p in sorted(glob.glob(exp + "/rollout_index/step*.tsv")):
    step = int(re.search(r"step(\d+)\.tsv", p).group(1)); g = collections.defaultdict(list); rows = []
    with open(p) as f:
        next(f)
        for line in f:
            _, row, grp, ds, drow, rew, length = line.rstrip("\n").split("\t")
            rew = float(rew); length = int(float(length)); rows.append(ds)
            for k in (ds, "ALL"):
                s = S[k]; s["n"] += 1; s["len"].append(length); s["rsum"] += rew
                if rew >= 1.0: s["r1"] += 1
                elif rew <= 0.0: s["r0"] += 1
                else: s["rp"] += 1
                if drow != "": s["rows"].add((ds, drow))
            g[(ds, grp)].append(rew)
    for (ds, grp), rs in g.items():
        c = sum(1 for r in rs if r >= 1.0)
        for k in (ds, "ALL"):
            s = S[k]; s["groups"] += 1
            if c == len(rs): s["g_all"] += 1
            elif c == 0: s["g_none"] += 1
            else: s["g_some"] += 1
    names[step] = rows
    with open("%s/train_data_step%d.jsonl" % (exp, step)) as f:
        for i, line in enumerate(f):
            r = json.loads(line); ds = rows[i]
            c = r["content"]; m = c[0] if isinstance(c, list) and c and isinstance(c[0], list) else c
            mask = r["sample_loss_mask"]; mask = mask[0] if isinstance(mask, list) else mask
            for k in (ds, "ALL"):
                s = S[k]; s["msgs"][len(m)] += 1; s["chars"] += sum(len(x) for x in m if isinstance(x, str))
                if not mask: s["masked"] += 1
def pct(a, q): return a[min(len(a) - 1, int(q * len(a)))]
res = {}
for k, s in S.items():
    a = sorted(s["len"]); n = s["n"]; turns = s["msgs"]
    res[k] = {"rollouts": n, "mean_reward": round(s["rsum"] / n, 4), "full": s["r1"], "zero": s["r0"], "partial": s["rp"], "masked": s["masked"],
              "tokens": {"mean": int(sum(a) / n), "p50": pct(a, 0.5), "p90": pct(a, 0.9), "p99": pct(a, 0.99), "max": a[-1], "sum": sum(a)},
              "at_limit": sum(1 for x in a if x >= LIMIT - 8), "over_16k": sum(1 for x in a if x > 16384),
              "distinct_prompts": len(s["rows"]), "groups": s["groups"], "groups_all_correct": s["g_all"], "groups_none_correct": s["g_none"], "groups_mixed": s["g_some"],
              "messages": {"mean": round(sum(m * c for m, c in turns.items()) / float(n), 2), "min": min(turns), "max": max(turns)},
              "chars_per_token": round(s["chars"] / float(sum(a)), 2)}
json.dump(res, open(out, "w"), indent=0)
for k in sorted(res): print(k, json.dumps(res[k]))
