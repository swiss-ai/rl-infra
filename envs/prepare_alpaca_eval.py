#!/usr/bin/env python3
"""Turn AlpacaEval 2.0 into the validation rows of envs/ultrafeedback-rubric.yaml.

Standard library only, so it runs on a login node:
    python3.12 envs/prepare_alpaca_eval.py $SCRATCH/rubric-judge/alpaca_eval

Its 805 prompts with the GPT-4 Turbo references of its default baseline. As in the training rows, the
few-shot prompt goes into each row for the policy, and the judge reads `question` and the reply.
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

from prepare_ultrafeedback import normalized, write_rows

URL = "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/gpt4_1106_preview/model_outputs.json"
AGENT = {"type": "responses_api_agents", "name": "alpaca_eval_simple_agent"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--few-shot", type=Path, default=Path(__file__).with_name("ultrafeedback-few-shot.txt"))
    args = parser.parse_args()

    template = args.few_shot.read_text().rstrip("\n")
    few_shot = {normalized(q) for q in re.findall(r"^User: (.+)$", template, flags=re.M)}
    with urllib.request.urlopen(URL, timeout=120) as response:
        references = json.load(response)

    overlap = [r["instruction"] for r in references if normalized(r["instruction"]) in few_shot]
    if overlap:
        raise SystemExit(f"few-shot examples are validation prompts: {overlap}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = (
        {
            # replace, not format: the examples and the prompts are free text with their own braces.
            "responses_create_params": {
                "input": [{"role": "user", "content": template.replace("{}", r["instruction"])}]
            },
            "question": r["instruction"],
            "baseline_answer": r["output"],
            "source": r["dataset"],
            "agent_ref": AGENT,
        }
        for r in references
    )
    write_rows(args.out_dir / "validation.jsonl", rows)
    print(f"{len(references)} validation rows")


if __name__ == "__main__":
    main()
