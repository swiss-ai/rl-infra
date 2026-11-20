"""Builds the evaluation sets for the STEM checkpoints in $BENCH_DIR (Python 3.6, data mover).
Benchmarks come from the Hugging Face datasets-server API; the four environment sets are held out from the data the
STEM runs train on. Each set is written as <name>.jsonl (input, output; the format the runs validate with) with its
repeats already in the file, plus <name>.meta.jsonl (one line per distinct item) for per-category results."""
import collections, hashlib, json, os, random, subprocess, sys, time
OUT = os.environ.get("BENCH_DIR", "/scratch/hanint06/data/benchmarks"); D = os.environ.get("STEM_DATA", "/scratch/hanint06/data/stem-rl")
# The rollout index of the run whose undrawn Open QA rows become the held-out set.
RUN_INDEX = os.environ.get("RUN_INDEX", "/scratch/hanint06/nemo-rl/logs/sftv02-stem-mix-run5-32k-49n/exp_001/rollout_index")
MATH = ("Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer "
        "(without quotes) where $Answer is the answer to the problem.\n\n%s\n\nRemember to put your answer on its own line after \"Answer:\".")
def rows(ds, split, config="default"):
    out, off = [], 0
    while True:
        url = "https://datasets-server.huggingface.co/rows?dataset=%s&config=%s&split=%s&offset=%d&length=100" % (ds, config, split, off)
        for attempt in range(6):
            try:
                r = json.loads(subprocess.check_output(["curl", "-s", "--max-time", "60", url]).decode("utf-8")); break
            except Exception:
                time.sleep(5 * (attempt + 1))
        if "rows" not in r: raise SystemExit("%s: %s" % (ds, str(r)[:300]))
        out += [x["row"] for x in r["rows"]]; off += 100
        if off >= r["num_rows_total"]: return out
def write(name, items, repeats):
    with open("%s/%s.jsonl" % (OUT, name), "w") as f:
        for _ in range(repeats):
            for it in items: f.write(json.dumps({"input": it["input"], "output": it["output"]}, ensure_ascii=False) + "\n")
    with open("%s/%s.meta.jsonl" % (OUT, name), "w") as f:
        for it in items: f.write(json.dumps({k: v for k, v in it.items() if k != "input"}, ensure_ascii=False) + "\n")
    print("%-26s %5d items x %2d = %6d rows" % (name, len(items), repeats, len(items) * repeats))
def key(t): return hashlib.md5(t.strip().encode("utf-8")).hexdigest()
which = sys.argv[1:] or ["aime", "math500", "mmlupro", "envs"]
if "aime" in which:
    for y in ("2025", "2026"):
        r = rows("MathArena/aime_" + y, "train")
        write("aime%s_x32" % y[2:], [{"input": MATH % x["problem"], "output": str(x["answer"]), "id": x.get("problem_idx")} for x in r], 32)
if "math500" in which:
    r = rows("HuggingFaceH4/MATH-500", "test")
    write("math500_x4", [{"input": MATH % x["problem"], "output": x["answer"], "id": x["unique_id"], "subject": x["subject"], "level": x["level"]} for x in r], 4)
if "mmlupro" in which:
    r = rows("TIGER-Lab/MMLU-Pro", "test"); L = "ABCDEFGHIJ"; items = []
    for x in r:
        n = len(x["options"]); letters = "/".join(L[:n])
        q = ("Answer the following multiple choice question. The last line of your response should be in the following format: "
             "\x27Answer: %s\x27 (e.g. \x27Answer: %s\x27).\n\n%s\n\n%s") % (letters, L[n - 1], x["question"], "\n".join("%s. %s" % (L[i], o) for i, o in enumerate(x["options"])))
        items.append({"input": q, "output": x["answer"], "id": x["question_id"], "category": x["category"], "options": n})
    write("mmlu_pro_x1", items, 1)
if "envs" in which:
    rnd = random.Random(20261118)
    # Reasoning Gym: the held-out file prepare.py wrote next to the training file (83 tasks).
    rg = [json.loads(l) for l in open(D + "/reasoning_gym/validation.jsonl")]
    train_rg = set(key(json.loads(l)["input"]) for l in open(D + "/reasoning_gym/train.jsonl"))
    print("reasoning_gym validation rows also in train:", sum(key(x["input"]) in train_rg for x in rg))
    write("reasoning_gym_val_x8", [dict(x, task=json.loads(x["output"]).get("task")) for x in rg], 8)
    # ScienceWorld: the episodes of the dev variations (the training file uses train variations only).
    sw = [l for l in open(D + "/scienceworld/sciworld_act_dev.jsonl")]
    with open(OUT + "/sciworld_act_dev_x4.jsonl", "w") as f:
        for _ in range(4): f.writelines(sw)
    print("%-26s %5d items x %2d = %6d rows (copied as is: ScienceWorld rows carry messages and episode settings)" % ("sciworld_act_dev_x4", len(sw), 4, 4 * len(sw)))
    # Open QA: the gold pool lies wholly inside the training file, so the held-out questions are rows of the training
    # file that run 5 at 32K never drew (rollout index); later runs must leave them out of training to keep them held out.
    import glob as _g
    used = set()
    for p in _g.glob(RUN_INDEX + "/step*.tsv"):
        for l in open(p):
            c = l.rstrip("\n").split("\t")
            if c[3] == "openqa-openqa_science" and c[4] != "": used.add(int(c[4]))
    sci = [json.loads(l) for l in open(D + "/openqa/openqa_science.jsonl")]
    smeta = [json.loads(l) for l in open(D + "/openqa/openqa_science_meta.jsonl")]
    free = [i for i in range(len(sci)) if i not in used]
    print("openqa training rows %d, drawn by the run %d, never drawn %d" % (len(sci), len(used), len(free)))
    pick = sorted(rnd.sample(free, 500))
    write("openqa_val_x2", [dict(sci[i], file_row=i, domain=smeta[i].get("domain"), kind=smeta[i].get("kind")) for i in pick], 2)
    # Enigmata: pool puzzles not in the training file, 10 percent of the training count of each (task, difficulty).
    train_en = [json.loads(l) for l in open(D + "/enigmata/train/enigmata_v1.jsonl")]
    used_en = set(key(x["input"]) for x in train_en)
    want = collections.Counter((x["task"], x["difficulty"]) for x in train_en)
    pool = collections.defaultdict(list)
    for p in _g.glob(D + "/enigmata/pool/*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            if key(r["prompt"]) not in used_en: pool[(r["task"], r["difficulty"])].append(r)
    items = []
    for (t, d), n in sorted(want.items()):
        k = max(1, int(round(n * 0.1))); cand = pool.get((t, d), [])
        for r in rnd.sample(cand, min(k, len(cand))):
            items.append({"input": r["prompt"], "output": json.dumps({"task": r["task"], "difficulty": r["difficulty"], "answer": r["answer"], "meta": r["meta"]}, ensure_ascii=False), "task": r["task"], "difficulty": r["difficulty"]})
    print("enigmata: %d (task, difficulty) buckets, %d tasks" % (len(want), len(set(t for t, _ in want))))
    with open(OUT + "/enigmata_val_x2.jsonl", "w") as f:
        for _ in range(2):
            for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("%-26s %5d items x %2d = %6d rows" % ("enigmata_val_x2", len(items), 2, 2 * len(items)))
