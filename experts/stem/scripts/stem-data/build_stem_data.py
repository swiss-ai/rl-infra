"""Builds the STEM-mix RL data files (input/output JSONL for NeMo-RL's ResponseDataset).

Outputs in OUT:
  numina_run1.jsonl  17,280 NuminaMath-1.5 competition problems (Run 1: 64 per step x 270)
  numina_run2.jsonl  first 10,800 of the same order (Run 2: 40 per step x 270)
  mcqa_run2.jsonl    6,480 Nemotron-RL-knowledge-mcqa questions (Run 2: 24 per step x 270)
  gpqa_diamond_x8.jsonl  198 GPQA-Diamond questions, each 8 times (validation, mean@8)
  summary.json       counts at every filter
Math problems use DAPO-Math-17K's own prompt wrapper, so they look exactly like the DAPO prompts.
"""
import glob, hashlib, json, logging, os, random, re, sys

import pandas as pd

DL = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
logging.getLogger("math_verify").setLevel(logging.CRITICAL)
from math_verify import parse
from math_verify.parser import LatexExtractionConfig

summary = {}
DAPO_HEAD = ("Solve the following math problem step by step. The last line of your response should be of the "
             "form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n")
DAPO_TAIL = "\n\nRemember to put your answer on its own line after \"Answer:\"."
MCQ_HEAD = ("Answer the following multiple choice question. The last line of your response should be in the "
            "following format: 'Answer: A/B/C/D' (e.g. 'Answer: D').\n\n")


def norm(t):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t.lower()).split())


def shingles(t, n=8):
    w = norm(t).split()
    if len(w) < n:
        return {hashlib.md5(" ".join(w).encode()).hexdigest()[:16]} if w else set()
    return {hashlib.md5(" ".join(w[i:i + n]).encode()).hexdigest()[:16] for i in range(len(w) - n + 1)}


class Index:
    """Flags texts whose 8-word shingles are at least half contained in an indexed text."""

    def __init__(self):
        self.sh = set()
        self.exact = set()

    def add(self, t):
        self.sh |= shingles(t)
        self.exact.add(norm(t))

    def hit(self, t, frac=0.5):
        if norm(t) in self.exact:
            return True
        s = shingles(t)
        return bool(s) and len(s & self.sh) / len(s) >= frac


def dapo_problem(content):
    c = content
    if c.startswith(DAPO_HEAD):
        c = c[len(DAPO_HEAD):]
    if c.endswith(DAPO_TAIL):
        c = c[: -len(DAPO_TAIL)]
    return c


# ---- reference sets: DAPO-17K (dedup) and the math evals (decontamination) ----
dapo = pd.read_parquet(f"{DL}/BytedTsinghua-SIA__DAPO-Math-17k/data/dapo-math-17k.parquet")
dapo_contents = [p[0]["content"] for p in dapo["prompt"]]
summary["dapo_rows"] = len(dapo_contents)
summary["dapo_unique_prompts"] = len(set(dapo_contents))
summary["dapo_wrapper_matches"] = sum(c.startswith(DAPO_HEAD) and c.endswith(DAPO_TAIL) for c in dapo_contents)
dapo_index = Index()
for c in set(dapo_contents):
    dapo_index.add(dapo_problem(c))

eval_index = Index()
eval_counts = {}
aime24 = pd.read_parquet(f"{DL}/BytedTsinghua-SIA__AIME-2024/data/aime-2024.parquet")
for p in aime24["prompt"]:
    eval_index.add(dapo_problem(p[0]["content"]))
eval_counts["aime24"] = len(aime24)
for name in ["aime_2025", "aime_2026", "hmmt_feb_2025", "hmmt_feb_2026"]:
    df = pd.read_parquet(f"{DL}/MathArena__{name}/data/train-00000-of-00001.parquet")
    for p in df["problem"]:
        eval_index.add(p)
    eval_counts[name] = len(df)
summary["eval_problems_indexed"] = eval_counts

# ---- NuminaMath-1.5 competition subset ----
num = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DL}/project-numina__NuminaMath-1.5/data/*.parquet"))],
                ignore_index=True)
summary["numina_rows"] = len(num)
SOURCES = {"olympiads", "aops_forum", "cn_contest", "amc_aime", "inequalities", "number_theory"}
steps = {}
m = num[num["source"].isin(SOURCES)]; steps["competition_sources"] = len(m)
m = m[m["question_type"] == "math-word-problem"]; steps["word_problems"] = len(m)
m = m[(m["problem_is_valid"] == "Yes") & (m["solution_is_valid"] == "Yes")]; steps["valid_problem_and_solution"] = len(m)
m = m[m["synthetic"].astype(str) != "True"]; steps["not_synthetic"] = len(m)
ans = m["answer"].fillna("").astype(str).str.strip()
m = m[(ans != "") & (~ans.isin(["proof", "notfound"])) & (ans.str.len() <= 40) & (~ans.str.contains(";|\\\\text", regex=True))]
steps["short_checkable_answer"] = len(m)
FIG = re.compile(r"\b(figure|fig\.|diagram|picture|image|shown below|graph below)\b", re.I)
m = m[(m["problem"].str.len() <= 2500) & (m["problem"].str.len() >= 40) & (~m["problem"].str.contains(FIG))]
steps["no_figure_and_length_ok"] = len(m)
gold_cfg = (LatexExtractionConfig(),)
ok = []
for a in m["answer"].astype(str):
    try:
        ok.append(len(parse("\\boxed{" + a.strip() + "}", extraction_config=gold_cfg)) > 0)
    except Exception:
        ok.append(False)
m = m[ok]; steps["answer_parses_with_math_verify"] = len(m)
m = m.assign(_n=m["problem"].map(norm)).drop_duplicates("_n"); steps["deduplicated"] = len(m)
keep_dapo = [not dapo_index.hit(p) for p in m["problem"]]
steps["overlap_with_dapo_removed"] = len(keep_dapo) - sum(keep_dapo)
m = m[keep_dapo]
keep_eval = [not eval_index.hit(p) for p in m["problem"]]
steps["overlap_with_evals_removed"] = len(keep_eval) - sum(keep_eval)
m = m[keep_eval]; steps["final_pool"] = len(m)
summary["numina_filter_steps"] = steps
summary["numina_pool_by_source"] = m["source"].value_counts().to_dict()
rows = [{"input": DAPO_HEAD + p.strip() + DAPO_TAIL, "output": a.strip(), "source": s}
        for p, a, s in zip(m["problem"], m["answer"].astype(str), m["source"])]
random.Random(42).shuffle(rows)
for name, n in [("numina_run1", 17280), ("numina_run2", 10800)]:
    assert len(rows) >= n, (name, len(rows))
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in rows[:n]:
            f.write(json.dumps({"input": r["input"], "output": r["output"]}) + "\n")
summary["numina_run1_by_source"] = pd.Series([r["source"] for r in rows[:17280]]).value_counts().to_dict()

# ---- GPQA-Diamond validation (and its index, for MCQ decontamination) ----
g = pd.read_parquet(f"{DL}/fingertap__GPQA-Diamond/test/gpqa_diamond.parquet")
summary["gpqa_rows"] = len(g)
summary["gpqa_answer_letters"] = g["answer"].value_counts().to_dict()
gpqa_index = Index()
with open(f"{OUT}/gpqa_diamond_x8.jsonl", "w") as f:
    for q, a in zip(g["question"], g["answer"]):
        gpqa_index.add(q.split("\nA. ")[0])
        for _ in range(8):
            f.write(json.dumps({"input": MCQ_HEAD + q.strip(), "output": a.strip()}) + "\n")

# ---- Nemotron-RL-knowledge-mcqa ----
mc = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DL}/nvidia__Nemotron-RL-knowledge-mcqa/data/train-*.parquet"))],
               ignore_index=True)
summary["mcqa_rows"] = len(mc)
contents = [c["input"][0]["content"] for c in mc["responses_create_params"]]
nopt = [sum(1 for o in opts if any(v is not None for v in o.values())) for opts in mc["options"]]
pass_rate = [p[0]["pass_rate"] if len(p) else None for p in mc["reward_profiles"]]
summary["mcqa_options"] = pd.Series(nopt).value_counts().head(8).to_dict()
summary["mcqa_qwen3_30b_pass_rate"] = pd.Series([round(x, 1) if x is not None else -1 for x in pass_rate]).value_counts().sort_index().to_dict()
summary["mcqa_example_tail"] = contents[0][-400:]
keep = []
for i, (c, n, a) in enumerate(zip(contents, nopt, mc["expected_answer"])):
    if n < 4 or len(c) > 3500 or not isinstance(a, str) or not re.fullmatch(r"[A-Z]", a.strip()):
        continue
    keep.append(i)
summary["mcqa_after_format_filters"] = len(keep)
k2 = [i for i in keep if not gpqa_index.hit(contents[i])]
summary["mcqa_overlap_with_gpqa_removed"] = len(keep) - len(k2)
random.Random(42).shuffle(k2)
with open(f"{OUT}/mcqa_run2.jsonl", "w") as f:
    for i in k2[:6480]:
        f.write(json.dumps({"input": contents[i].strip(), "output": mc["expected_answer"][i].strip()}) + "\n")
sel = k2[:6480]
summary["mcqa_run2_options"] = pd.Series([nopt[i] for i in sel]).value_counts().to_dict()
summary["mcqa_run2_pass_rate"] = pd.Series([round(pass_rate[i], 1) if pass_rate[i] is not None else -1 for i in sel]).value_counts().sort_index().to_dict()
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
