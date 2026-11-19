#!/usr/bin/env python3
"""Prompts and rollouts of a run of this task in the schema of the Apertus 2 experts handover
(apertus-program docs/engineering/models/apertus-2/apertus2-experts.md), as for the Lean expert of round 1.
    python3 handover_lean.py --run <RUN> --job <training job> --train <rows> --heldout <rows> --steps <n> <entry>
  --train    the Gym rows the run trained on (e.g. $LEAN_DATA/data/goedel-train-s1.v434.jsonl)
  --heldout  Gym rows never trained on
  --steps    the training steps to write (the run's last trained step)
Writes <entry>/prompts/{train,heldout}.jsonl and <entry>/rollouts/step<NNN>.jsonl from
$SCRATCH/nemo-rl/results/<RUN>/rollouts/train_weights<v>.jsonl and the job's driver log.
- Prompts: `id` (<source>-<line>), `source`, `conversation` (Apertus 2 native, schema_version 2, the run's frame:
  thinking medium, no tools), `meta` (name, family, goedel_index, header, formal_statement, gym_splits_statement).
- Rollouts: `prompt_id`, `step`, `sample` (0-15), `messages` (user prompt; assistant = vLLM's raw generation_str with
  the think block; environment = Kimina's proof_status and Lean output), `reward`, `length` (generated tokens).
- Steps (asynchronous GRPO, trajectory age 1): steps 1 and 2 trained on weight version 0, whose file holds both batches
  interleaved; they are split by the collector's log lines "Buffered prompt group (_ng_task_index=.., target_weight=..)"
  (target 0 = step 1). Step s >= 3 trained on version s-2. Each step's mean reward is checked against the
  driver log's `Avg Reward` for that step.
"""
import argparse
import json
import os
import re

FRAME = {"identity": None, "thinking": "medium", "behavior": None, "tools": [], "environment": None}


def user_text(row):
    return row["responses_create_params"]["input"][0]["content"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--job", required=True, type=int)
    ap.add_argument("--train", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--steps", required=True, type=int)
    ap.add_argument("--group", type=int, default=16, help="rollouts per prompt")
    ap.add_argument("--prompts-per-step", type=int, default=32)
    ap.add_argument("entry")
    a = ap.parse_args()
    S = os.environ["SCRATCH"]
    run_dir = f"{S}/nemo-rl/results/{a.run}"
    driver = f"{S}/nemo-rl/slurm/{a.job}-logs/ray-driver.log"
    os.makedirs(f"{a.entry}/prompts", exist_ok=True)
    os.makedirs(f"{a.entry}/rollouts", exist_ok=True)

    def convert(src, dst, source):
        ids = {}
        with open(dst + ".part", "w", encoding="utf-8") as out:
            for line_no, line in enumerate(open(src, encoding="utf-8")):
                r = json.loads(line)
                pid = f"{source}-{line_no:06d}"
                text = user_text(r)
                assert text not in ids, f"duplicate prompt in {src}"
                ids[text] = pid
                meta = {k: r[k] for k in ("name", "source", "goedel_index", "header", "formal_statement",
                                          "gym_splits_statement") if k in r}
                meta["family"] = meta.pop("source", None)
                out.write(json.dumps({"id": pid, "source": source,
                                      "conversation": {"schema_version": 2, "system": FRAME,
                                                       "items": [{"type": "user", "payload": text}]},
                                      "meta": meta}, ensure_ascii=False) + "\n")
        os.rename(dst + ".part", dst)
        print(f"{dst}: {len(ids)} prompts")
        return ids

    train_ids = convert(a.train, f"{a.entry}/prompts/train.jsonl", "goedel_rl_v2_train")
    convert(a.heldout, f"{a.entry}/prompts/heldout.jsonl", "goedel_rl_v2_heldout")

    logged, block, targets = [], False, []
    n_first = 2 * a.prompts_per_step
    for line in open(driver, encoding="utf-8", errors="replace"):
        if "Training Results" in line:
            block = True
        elif block and "Avg Reward:" in line:
            logged.append(float(line.split("Avg Reward:")[1].split()[0]))
            block = False
        m = re.search(r"Buffered prompt group \(_ng_task_index=\d+, target_weight=(\d+)\)", line)
        if m and len(targets) < n_first:
            targets.append(int(m[1]))
    print(f"driver log: {len(logged)} training steps")

    def rollout(r, step, sample):
        gen = r["response"]["output"][0].get("generation_str")
        gen = json.loads(gen) if isinstance(gen, str) and gen.startswith('"') else gen
        env = f"proof_status: {r.get('proof_status')}"
        stdout = (r.get("compiler_output") or {}).get("stdout")
        if stdout:
            env += "\n" + stdout
        usage = r["response"].get("usage") or {}
        return {"prompt_id": train_ids[user_text(r)], "step": step, "sample": sample,
                "messages": [{"role": "user", "content": user_text(r)},
                             {"role": "assistant", "content": gen},
                             {"role": "environment", "content": env}],
                "reward": float(r["reward"]), "length": usage.get("output_tokens")}

    def write_step(step, rows):
        seen, out = {}, []
        for r in rows:
            pid = train_ids[user_text(r)]
            seen[pid] = seen.get(pid, -1) + 1
            out.append(rollout(r, step, seen[pid]))
        assert len(seen) == a.prompts_per_step and all(n == a.group - 1 for n in seen.values()), f"step {step}"
        path = f"{a.entry}/rollouts/step{step:03d}.jsonl"
        with open(path + ".part", "w", encoding="utf-8") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        os.rename(path + ".part", path)
        mean = sum(o["reward"] for o in out) / len(out)
        return mean, (abs(mean - logged[step - 1]) < 1e-3 if step <= len(logged) else None)

    def load(v):
        return [json.loads(line) for line in open(f"{run_dir}/rollouts/train_weights{v}.jsonl", encoding="utf-8")]

    v0 = load(0)
    groups = [v0[i:i + a.group] for i in range(0, len(v0), a.group)]
    assert len(groups) == n_first and all(len({user_text(r) for r in g}) == 1 for g in groups), "version 0 layout"
    assert sorted(targets) == [0] * a.prompts_per_step + [1] * a.prompts_per_step, f"targets {targets}"
    checks = []
    for step, target in ((1, 0), (2, 1)):
        rows = [r for g, t in zip(groups, targets) if t == target for r in g]
        checks.append((step, *write_step(step, rows)))
    for step in range(3, a.steps + 1):
        checks.append((step, *write_step(step, load(step - 2))))
    bad = [c for c in checks if c[2] is not True]
    print(f"rollouts: {len(checks)} steps written; mean reward equals the driver log for {len(checks) - len(bad)}")
    for c in bad:
        print(f"  MISMATCH step {c[0]}: file mean {c[1]:.4f}")


if __name__ == "__main__":
    main()
