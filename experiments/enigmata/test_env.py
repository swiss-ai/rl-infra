"""Checks of the native Enigmata environment (NeMo-RL nemo_rl/environments/enigmata_environment.py).

Runs in the NeMo-RL container on one node with a small Ray instance; no model, no GPU work.
Apertus 2 outputs are built directly as Apertus2Output objects.

  python3 test_env.py <NeMo-RL dir> <Enigmata checkout> <pool dir>

Checks:
  1. For every task: the stored answer, written as the prompt's example format (grid lines,
     list or string in a code block), in the claim. Reports which tasks score 1 this way
     (a plumbing check, not every task's format is reproduced exactly).
  2. Every task, garbage and very long text in the claim: score 0, no verifier exception.
  3. Empty claim with the answer in the reply: graded on the reply (fallback_frame reply).
  4. A verifier that sleeps past timeout_s: status timeout, reward 0, the next call works.
  5. A verifier that raises: status error:<type>, reward 0.
  6. Unparsed output (error set): reward 0 without a call.
"""

import json
import os
import shutil
import sys
import tempfile
import time

nemo_rl_dir, repo_dir, pool_dir = sys.argv[1:4]
sys.path.insert(0, nemo_rl_dir)
# Ray's worker processes inherit the environment of the local cluster ray.init() starts.
os.environ["PYTHONPATH"] = nemo_rl_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

import ray  # noqa: E402

from nemo_rl.data.apertus2_format import Apertus2Output  # noqa: E402
from nemo_rl.environments.enigmata_environment import EnigmataEnvironment  # noqa: E402

# A copy of the task tree with two extra verifiers for the timeout and error checks.
tmp = tempfile.mkdtemp(prefix="enigmata_test_")
tasks_src = os.path.join(repo_dir, "verifiable_tasks", "tasks")
tasks_dst = os.path.join(tmp, "verifiable_tasks", "tasks")
os.makedirs(tasks_dst)
for t in os.listdir(tasks_src):
    os.symlink(os.path.join(tasks_src, t), os.path.join(tasks_dst, t))
for name, body in {
    "zz_sleepy": "import time\ndef verify(pred, answer, meta):\n    time.sleep(30)\n    return 1\n",
    "zz_broken": "def verify(pred, answer, meta):\n    raise KeyError('boom')\n",
    "zz_swallow": (
        "import time\ndef verify(pred, answer, meta):\n    try:\n        time.sleep(30)\n"
        "    except Exception:\n        return 1\n    return 1\n"
    ),
}.items():
    os.makedirs(os.path.join(tasks_dst, name))
    with open(os.path.join(tasks_dst, name, "verifier.py"), "w") as f:
        f.write(body)

ray.init(num_cpus=32, include_dashboard=False)
cfg = {
    "repo_dir": tmp,
    "num_workers": 4,
    "reward_frame": "claim",
    "fallback_frame": "reply",
    "timeout_s": 3.0,
    "hard_timeout_s": 60.0,
    "log_every": 50,
    "stats_path": os.path.join(tmp, "stats.jsonl"),
    "stats_text_chars": 80,
}
env = EnigmataEnvironment.remote(cfg)


def out(claim=None, reply=None, error=None, truncated=False):
    return Apertus2Output(think="", claim=claim, reply=reply, calls=[], error=error, truncated=truncated)


def step(rows, outputs):
    logs = [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "apertus2_output": o}]
            for o in outputs]
    meta = [{"ground_truth": json.dumps({"task": r["task"], "difficulty": r["difficulty"],
                                         "answer": r["answer"], "meta": r["meta"]})} for r in rows]
    t0 = time.perf_counter()
    res = ray.get(env.step.remote(logs, meta, True))
    return res, time.perf_counter() - t0


def as_text(answer):
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except ValueError:
            return answer
    if isinstance(answer, list) and answer and all(isinstance(r, list) for r in answer) and all(
        all(isinstance(x, (int, float)) for x in r) for r in answer
    ):
        return "\n".join(" ".join(str(x) for x in r) for r in answer)
    return json.dumps(answer) if not isinstance(answer, str) else answer


rows = {}
for fn in sorted(os.listdir(pool_dir)):
    if fn.endswith(".jsonl"):
        with open(os.path.join(pool_dir, fn)) as f:
            rows[fn[:-6]] = json.loads(f.readline())

print("== 1. stored answer in a code block, in the claim")
tasks = sorted(rows)
res, dt = step([rows[t] for t in tasks], [out(claim="```\n" + as_text(rows[t]["answer"]) + "\n```") for t in tasks])
for t, r in zip(tasks, res.rewards.tolist()):
    print(f"  {t}: {r}")
print(f"  {len(tasks)} samples in {dt:.2f} s")

print("== 2. garbage and long text")
garbage = ["hello", "```\nnot an answer\n```", "[1, 2, 3]", "x" * 200000, "```" * 5000, "\n" * 1000]
batch, outs = [], []
for t in tasks:
    for g in garbage:
        batch.append(rows[t])
        outs.append(out(claim=g))
res, dt = step(batch, outs)
print(f"  max reward {max(res.rewards.tolist())}, {len(batch)} samples in {dt:.2f} s")

print("== 3. empty claim, answer in reply")
res, _ = step([rows[t] for t in tasks],
              [out(claim=None, reply="Here it is:\n```\n" + as_text(rows[t]["answer"]) + "\n```\n") for t in tasks])
print("  rewards:", dict(zip(tasks, res.rewards.tolist())))
print("  frame metrics sample:", res.metadata[0]["frame_metrics"])

print("== 4/5. timeout, error, swallowed timeout")
special = [dict(rows[tasks[0]], task=name) for name in ("zz_sleepy", "zz_broken", "zz_swallow")]
res, dt = step(special, [out(claim="x"), out(claim="x"), out(claim="x")])
print(f"  rewards {res.rewards.tolist()} in {dt:.2f} s (expect 0, 0, 0 after about 3 s)")
res, dt = step([rows[tasks[0]]], [out(claim="```\n" + as_text(rows[tasks[0]]["answer"]) + "\n```")])
print(f"  next call: reward {res.rewards.tolist()} in {dt:.2f} s")

print("== 6. unparsed output")
res, _ = step([rows[tasks[0]]], [out(error="bad frame", truncated=True)])
print(f"  reward {res.rewards.tolist()}")

ray.get(env.shutdown.remote())
print("== stats file (last 6 lines)")
with open(cfg["stats_path"]) as f:
    lines = f.readlines()
statuses = {}
for line in lines:
    r = json.loads(line)
    for k in ("primary_status", "side_status"):
        statuses[r[k]] = statuses.get(r[k], 0) + 1
print(f"  {len(lines)} lines; statuses {statuses}")
for line in lines[-6:]:
    print("  " + line.rstrip()[:300])
shutil.rmtree(tmp)
