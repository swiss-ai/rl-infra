#!/usr/bin/env python3
"""Turn nvidia/Nemotron-RL-ReasoningGym-v1 into the rows of envs/reasoning-gym-sft.yaml.

Standard library only, so it runs on a login node:
    python3.12 envs/prepare_reasoning_gym.py $SCRATCH/reasoning-gym/native

The rows are scored natively by NeMo-RL's math environment with verifier_type reasoning_gym (NeMo-RL
patch 0034): `input` is the task's prompt as the dataset gives it, and `output` is the ground truth as a
JSON object {"task", "question", "answer", "metadata"}, the entry the task's reasoning_gym scorer reads.

The dataset has 15,000 rows over 104 tasks, all at default difficulty, CC BY 4.0. Left out:
- gsm_symbolic: its generators were written by a closed model (Claude 3.5 Sonnet);
- arc_agi, rearc: ARC grids, kept out of training against the ARC evaluations;
- boxnet, graph_color, jugs, propositional_logic, rubiks_cube, rush_hour, word_ladder: the dataset's own gold
  answers score below 1 with the task's scorer; game_of_life_halting rewards wrong answers;
- the nine tasks whose scorers eval() or parse_expr the answer (the verifier refuses them as well);
- rows with an empty gold answer (one in base_conversion, whose scorer rejects it, and one in ab, which the
  verifier cannot score: an empty claim gets 0).
That leaves 84 tasks. Validation then train are sampled round-robin over the tasks (seed 0), so both are
balanced across tasks and disjoint. Every kept gold answer scores 1 with reasoning-gym 0.1.25 (check with
backends/nemo-rl/apertus2/check_rg_verifier.sh).
"""

import argparse
import collections
import json
import random
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1/resolve/main/data/train.jsonl"
DROP = {
    "gsm_symbolic", "arc_agi", "rearc",
    "boxnet", "graph_color", "jugs", "propositional_logic", "rubiks_cube", "rush_hour", "word_ladder",
    "game_of_life_halting",
    "n_queens", "spiral_matrix", "binary_matrix", "string_insertion", "countdown", "puzzle24",
    "simple_integration", "intermediate_integration", "polynomial_multiplication",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--validation", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with urllib.request.urlopen(URL) as f:
        rows = [json.loads(line) for line in f]
    by_task = collections.defaultdict(list)
    for r in rows:
        task = r["metadata"]["source_dataset"]
        if task not in DROP and str(r["answer"]).strip():
            by_task[task].append(r)
    tasks = sorted(by_task)
    rng = random.Random(args.seed)
    for task in tasks:
        rng.shuffle(by_task[task])

    cursor = {task: 0 for task in tasks}

    def take(n: int) -> list:
        picked, i = [], 0
        while len(picked) < n:
            task = tasks[i % len(tasks)]
            if cursor[task] < len(by_task[task]):
                picked.append(by_task[task][cursor[task]])
                cursor[task] += 1
            i += 1
        return picked

    validation = take(args.validation)
    train = take(args.train)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("validation", validation)):
        with open(args.out_dir / f"{name}.jsonl", "w") as f:
            for r in split:
                truth = {
                    "task": r["metadata"]["source_dataset"],
                    "question": r["question"],
                    "answer": r["answer"],
                    "metadata": r["metadata"],
                }
                prompt = r["responses_create_params"]["input"][-1]["content"].rstrip()
                f.write(json.dumps({"input": prompt, "output": json.dumps(truth), "uuid": r["uuid"]}) + "\n")
    print(f"{len(tasks)} tasks; {len(train)} train and {len(validation)} validation rows in {args.out_dir}")


if __name__ == "__main__":
    main()
