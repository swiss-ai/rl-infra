"""Judge quality on the real triples (hangang-dm, Python 3.6). Reads the benchmark logs of each judge model and
compares modes and judges: verdict coverage, accuracy against math-verify on the rule-checkable triples, pairwise
agreement, agreement with the majority of all judges and modes. Writes quality.json and disagreements.txt."""
import itertools, json, os, random, sys
OUT = sys.argv[1]
MODELS = ["gemma", "qwen", "nemotron-a2"]
FILES = {"verdict": "bench_1_verdict_c1024.jsonl", "justify": "bench_3_justify_c1024.jsonl"}
base = "/scratch/hanint06/nemo-rl/logs/openqa-bench-%s"
triples, J = None, {}
for m in MODELS:
    d = base % m
    if not os.path.exists(os.path.join(d, "triples_used.jsonl")):
        continue
    t = [json.loads(l) for l in open(os.path.join(d, "triples_used.jsonl"))]
    if triples is None:
        triples = t
    assert len(t) == len(triples) and all(a["answer"] == b["answer"] for a, b in zip(t, triples))
    for mode, f in FILES.items():
        p = os.path.join(d, f)
        if os.path.exists(p):
            r = [json.loads(l) for l in open(p)]
            J["%s-%s" % (m, mode)] = {x["i"]: x for x in r}
n = len(triples)
res = {"triples": n, "gold": sum(1 for t in triples if "rule" in t), "judges": {}}
hard = {k: {i: (None if v[i].get("score") is None else int(v[i]["score"] >= 0.5)) for i in v} for k, v in J.items()}
for k, v in J.items():
    cov = sum(1 for i in v if v[i].get("score") is not None)
    pos = sum(1 for i in v if hard[k][i] == 1)
    g = [(hard[k][i], int(triples[i]["rule"] >= 0.5)) for i in v if "rule" in triples[i] and hard[k][i] is not None]
    acc = sum(a == b for a, b in g) / max(1, len(g))
    fp = sum(a == 1 and b == 0 for a, b in g); fn = sum(a == 0 and b == 1 for a, b in g)
    res["judges"][k] = {"calls": len(v), "verdict_coverage": round(cov / max(1, len(v)), 4), "yes_rate": round(pos / max(1, cov), 4),
                        "gold_n": len(g), "gold_accuracy": round(acc, 4), "gold_false_yes": fp, "gold_false_no": fn,
                        "gold_rule_yes": sum(b for _, b in g)}
keys = sorted(J)
res["pairwise_agreement"] = {}
for a, b in itertools.combinations(keys, 2):
    both = [i for i in range(n) if hard[a].get(i) is not None and hard[b].get(i) is not None]
    res["pairwise_agreement"]["%s vs %s" % (a, b)] = round(sum(hard[a][i] == hard[b][i] for i in both) / max(1, len(both)), 4)
maj = {}
for i in range(n):
    votes = [hard[k].get(i) for k in keys if hard[k].get(i) is not None]
    if len(votes) >= 3:
        maj[i] = int(sum(votes) * 2 > len(votes)) if sum(votes) * 2 != len(votes) else None
res["agreement_with_majority"] = {k: round(sum(hard[k][i] == maj[i] for i in maj if maj[i] is not None and hard[k].get(i) is not None) /
                                           max(1, sum(1 for i in maj if maj[i] is not None and hard[k].get(i) is not None)), 4) for k in keys}
json.dump(res, open(os.path.join(OUT, "quality.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
random.seed(0)
dis = [i for i in range(n) if len(set(hard[k].get(i) for k in keys if hard[k].get(i) is not None)) > 1]
with open(os.path.join(OUT, "disagreements.txt"), "w") as f:
    f.write("%d triples where judges or modes disagree; 60 random ones\n\n" % len(dis))
    for i in random.sample(dis, min(60, len(dis))):
        t = triples[i]
        f.write("#%d [%s %s rule=%s]\nQ: %s\nREF: %s\nANS: %s\nVERDICTS: %s\n\n" % (
            i, t.get("domain"), t.get("kind"), t.get("rule"), t["question"][:400].replace("\n", " "), t["reference"][:300],
            t["answer"][:300].replace("\n", " "), {k: hard[k].get(i) for k in keys}))
print("disagreements", len(dis))
