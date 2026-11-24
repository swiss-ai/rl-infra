"""Step 2 of the openqa debug set (runs on hangang-dm, Python 3.6, json only).
Joins meta.jsonl with the domain labels and writes, in DIR:
  openqa_science.jsonl       input/output rows (the dataset's own prompt, the reference answer), seed-42 shuffle:
                             Qwen3-235B pass rate < 1.0, science + CS domains (no medicine, law, humanities, ...)
  openqa_science_meta.jsonl  the same rows' row id, uuid, domain, answer kind, gold flag, pass rates
  openqa_gold.jsonl          the rule-checkable subset (number/expression answers math-verify parses)
  gen_sample.jsonl (+_meta)  1,000 non-gold + 500 gold questions for the judge benchmark's candidate answers
  summary.json
"""
import collections, glob, json, random, sys
D = sys.argv[1]
KEEP = {"physics", "chemistry", "biology", "engineering", "earth_science", "math", "computer_science"}
lab = {}
for f in glob.glob(D + "/labels_*.jsonl"):
    for l in open(f):
        r = json.loads(l); lab[r["row"]] = r["label"]
rows, s = [], collections.Counter()
dom_all = collections.Counter(lab.values())
for l in open(D + "/meta.jsonl"):
    r = json.loads(l)
    s["rows"] += 1
    p = r["pass_q235"]
    if p is None or p >= 1.0:
        s["dropped_q235_pass_1"] += 1; continue
    d = lab.get(r["row"], "MISSING"); r["domain"] = d
    if d not in KEEP:
        s["dropped_domain"] += 1; continue
    rows.append(r)
random.Random(42).shuffle(rows)
def dump(name, rs, meta_only=False):
    with open(D + "/" + name, "w") as f:
        for r in rs:
            f.write(json.dumps({"input": r["prompt"], "output": r["answer"]}) + "\n")
    with open(D + "/" + name.replace(".jsonl", "_meta.jsonl"), "w") as f:
        for r in rs:
            f.write(json.dumps({k: r[k] for k in ("row", "uuid", "domain", "kind", "gold", "pass_q235", "pass_gptoss", "question", "answer")}) + "\n")
dump("openqa_science.jsonl", rows)
gold = [r for r in rows if r["gold"]]
dump("openqa_gold.jsonl", gold)
nongold = [r for r in rows if not r["gold"]]
dump("gen_sample.jsonl", nongold[:1000] + gold[:500])
summ = {"counts": dict(s), "kept": len(rows), "kept_by_domain": dict(collections.Counter(r["domain"] for r in rows)),
        "kept_by_kind": dict(collections.Counter(r["kind"] for r in rows)), "gold": len(gold),
        "labels_all_rows": dict(dom_all)}
json.dump(summ, open(D + "/summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
