#!/bin/bash
# Test of the math environment's reasoning_gym verifier (NeMo-RL patch 0034) before a training job:
# installs the PYDEPS_RG overlay, starts the real Ray worker and scores, for every row of a native
# Reasoning Gym file, the gold answer (plain, in <answer> tags, in \boxed{}), an empty answer and the
# gold answer of another row of the same task, plus known reward hacks (NaN on complex_arithmetic and
# number_sorting, "0" on coin_flip, an unsafe task) and answers that some scorers pay a little for (any
# sentence, any palindrome, any fraction). Fails if a gold answer scores below 1 on more than MAX_GOLD_FAIL
# rows (default 0), or if any other case scores above 0, except another row's answer when it is also right.
# Rows from envs/prepare_reasoning_gym.py. Runs as a one-node job, after setup.sh has applied patch 0034 to
# the clone (run.sh, or a DRY_RUN=1 of it, does):
#   PYDEPS_RG=1 PYDEPS=$SCRATCH/nemo-rl-apertus2/pydeps-rg CHECK_SCRIPT=check_rg_verifier.sh \
#       RG_FILE=$SCRATCH/reasoning-gym/native/train.jsonl bash backends/nemo-rl/apertus2/check_env_job.sh
set -euo pipefail

: "${RG_FILE:?RG_FILE: a native Reasoning Gym jsonl}"
bash "$(dirname "$0")/pydeps.sh"
export RG_FILE MAX_GOLD_FAIL="${MAX_GOLD_FAIL:-0}"

python3.12 - <<'EOF'
import ast, collections, json, os, random

import ray

import nemo_rl.environments.math_environment as me
from nemo_rl.environments.reasoning_gym_verifier import UNSAFE_TASKS, ReasoningGymVerifyWorker, extract_answer
from nemo_rl.environments.utils import ENV_REGISTRY

print("nemo_rl from", os.path.dirname(me.__file__))
assert me.MathEnvironment.WORKER_CLASS_DICT["reasoning_gym"] is ReasoningGymVerifyWorker
assert ENV_REGISTRY["reasoning_gym"]["actor_class_fqn"].endswith("MathEnvironment")
assert extract_answer("x \\boxed{\\dfrac{1}{8}} y") == "\\dfrac{1}{8}"
assert extract_answer("a <answer>1</answer> b <answer> 2 </answer>") == "2"
assert extract_answer("  ") is None

rows = [json.loads(l) for l in open(os.environ["RG_FILE"])]
by_task = collections.defaultdict(list)
for r in rows:
    by_task[json.loads(r["output"])["task"]].append(r)
rng = random.Random(0)

cases = []  # (kind, task, response, ground_truth)
for r in rows:
    gt = json.loads(r["output"])
    task, gold = gt["task"], str(gt["answer"])
    cases += [("gold", task, gold, r["output"]),
              ("gold_tag", task, f"Final: <answer>{gold}</answer>", r["output"]),
              ("gold_boxed", task, f"So \\boxed{{{gold}}}.", r["output"]),
              ("empty", task, "", r["output"])]
    others = [o for o in by_task[task] if str(json.loads(o["output"])["answer"]) != gold]
    if others:
        cases.append(("other", task, str(json.loads(rng.choice(others)["output"])["answer"]), r["output"]))
synthetic = lambda task, answer: json.dumps({"task": task, "question": "", "answer": answer, "metadata": {}})
cases.append(("hack", "countdown", "__import__('os')", synthetic("countdown", "1")))
cases.append(("hack", "coin_flip", "0", synthetic("coin_flip", "0.375")))
for r in by_task["complex_arithmetic"][:5]:
    cases += [("hack", "complex_arithmetic", a, r["output"]) for a in ("nan", "NaN + NaNi", "inf")]
for r in by_task["number_sorting"][:5]:
    n = len(ast.literal_eval(json.loads(r["output"])["answer"]))
    cases.append(("hack", "number_sorting", json.dumps([float("nan")] * n), r["output"]))
for task, answer in (("sentence_reordering", "x"), ("palindrome_generation", "a"), ("fraction_simplification", "1/2")):
    cases += [("format", task, answer, r["output"]) for r in by_task[task][:5]
              if str(json.loads(r["output"])["answer"]) != answer]

ray.init(num_cpus=8, include_dashboard=False, log_to_driver=False)
workers = [ReasoningGymVerifyWorker.remote() for _ in range(8)]
chunks = [cases[i::8] for i in range(8)]
outs = ray.get([w.verify.remote([c[2] for c in ch], [c[3] for c in ch], True) for w, ch in zip(workers, chunks)])
scored = []
for ch, (scores, answers) in zip(chunks, outs):
    scored += [(c[0], c[1], s, a) for c, s, a in zip(ch, scores, answers)]

stats = collections.defaultdict(list)
for kind, task, s, _ in scored:
    stats[kind].append(s)
for kind, v in stats.items():
    print(f"{kind:11s} n={len(v):5d} mean={sum(v)/len(v):.3f} ==1: {sum(x >= 1 for x in v):5d} ==0: {sum(x == 0 for x in v):5d}")
gold_fail = collections.Counter(t for k, t, s, _ in scored if k.startswith("gold") and s < 1)
other_hit = collections.Counter(t for k, t, s, _ in scored if k == "other" and s >= 1)
print("gold < 1 by task:", dict(gold_fail))
print("other row's answer scored 1, by task (duplicate answers or lenient scorers):", dict(other_hit))
bad = [(k, t, s) for k, t, s, _ in scored if k in ("empty", "hack", "format") and s > 0]
assert not bad, bad
assert all(s in (0.0, 1.0) for _, _, s, _ in scored), "reward is not binary"
assert sum(gold_fail.values()) <= int(os.environ["MAX_GOLD_FAIL"]), "gold answers below 1"
print("RG VERIFIER OK")
EOF
