#!/usr/bin/env python3
"""Turn the UltraFeedback prompts into the rows of envs/ultrafeedback-rubric.yaml.

Standard library only, so it runs on a login node:
    python3.12 envs/prepare_ultrafeedback.py $SCRATCH/rubric-judge/ultrafeedback

Runs train on train.jsonl only; validation.jsonl is the held-out split, and they validate on
AlpacaEval (envs/prepare_alpaca_eval.py). A Gym rollout does not pass through NeMo-RL's prompt_file, so the few-shot prompt goes into each
row here, and `question` keeps the bare request for the judge.
"""

import argparse
import json
import os
import random
import re
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/openbmb/UltraFeedback/resolve/main/{}.jsonl"
# Leaves out false_qa and truthful_qa: trick questions, not general requests.
SOURCES = ("sharegpt", "evol_instruct", "ultrachat", "flan")
AGENT = {"type": "responses_api_agents", "name": "rubric_judge_simple_agent"}


def normalized(prompt):
    """Compare prompts without case, punctuation or spacing."""
    return " ".join(re.sub(r"[^\w ]", " ", prompt.lower()).split())


def read_prompts(sources, max_chars):
    """Unique (source, prompt) pairs under max_chars, in file order."""
    seen, prompts, dropped_long = set(), [], 0
    for source in sources:
        with urllib.request.urlopen(URL.format(source), timeout=120) as lines:
            for line in lines:
                prompt = json.loads(line)["instruction"].strip()
                if not prompt or prompt in seen:
                    continue
                seen.add(prompt)
                if len(prompt) > max_chars:
                    dropped_long += 1
                    continue
                prompts.append((source, prompt))
    return prompts, dropped_long


def to_row(source, prompt, template):
    # replace, not format: the examples and the prompts are free text with their own braces.
    return {
        "responses_create_params": {"input": [{"role": "user", "content": template.replace("{}", prompt)}]},
        "question": prompt,
        "source": source,
        "agent_ref": AGENT,
    }


def write_rows(path, rows):
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--few-shot", type=Path, default=Path(__file__).with_name("ultrafeedback-few-shot.txt"))
    parser.add_argument("--sources", nargs="+", default=list(SOURCES))
    parser.add_argument("--validation-size", type=int, default=1000)
    # About 500 tokens: the few-shot prompt and the reply share max_total_sequence_length.
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    template = args.few_shot.read_text().rstrip("\n")
    if template.count("{}") != 1:
        raise SystemExit(f"{args.few_shot} must have exactly one {{}}")

    prompts, dropped_long = read_prompts(args.sources, args.max_chars)
    # A prompt the policy sees as a few-shot example must not also be a task.
    few_shot = {normalized(q) for q in re.findall(r"^User: (.+)$", template, flags=re.M)}
    prompts = [p for p in prompts if normalized(p[1]) not in few_shot]
    random.Random(args.seed).shuffle(prompts)
    validation, train = prompts[: args.validation_size], prompts[args.validation_size :]
    held_out = {normalized(p[1]) for p in validation}
    train = [p for p in train if normalized(p[1]) not in held_out]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "train.jsonl", (to_row(*p, template) for p in train))
    write_rows(args.out_dir / "validation.jsonl", (to_row(*p, template) for p in validation))
    print(f"{len(train)} train, {len(validation)} held out, {dropped_long} over {args.max_chars} chars dropped")


if __name__ == "__main__":
    main()
