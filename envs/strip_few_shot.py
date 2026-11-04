#!/usr/bin/env python3
"""Rewrites prepared Gym rows for a checkpoint that needs no few-shot prompt.

The prompts stay the same rows in the same order, so runs on the pretraining checkpoint and on an SFT
checkpoint train and validate on the same questions. Each row's `question` becomes its only message.
    python3.12 envs/strip_few_shot.py $SCRATCH/rubric-judge/ultrafeedback $SCRATCH/rubric-judge/ultrafeedback-bare
    python3.12 envs/strip_few_shot.py $SCRATCH/rubric-judge/alpaca_eval $SCRATCH/rubric-judge/alpaca_eval-bare
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for source in args.source.glob("*.jsonl"):
        path = args.out_dir / source.name
        tmp = path.with_suffix(f".tmp.{os.getpid()}")
        with source.open() as rows, tmp.open("w") as out:
            for line in rows:
                row = json.loads(line)
                row["responses_create_params"]["input"] = [{"role": "user", "content": row["question"]}]
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
        print(path)


if __name__ == "__main__":
    main()
