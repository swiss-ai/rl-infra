#!/usr/bin/env python3
"""AIME 2024 on a served checkpoint, with the prompts of the GRPO validation set (DAPOMathAIME2024).

Run it with the venv from setup.sh, on the login node:
    venv/bin/python aime.py prompts --repeat 4 > aime.jsonl
    venv/bin/python chat.py --prompts aime.jsonl --workers 120 --temperature 1 --top-p 1 --max-tokens 16000
    venv/bin/python aime.py grade results/<file>.jsonl

Answers are integers, so grading is exact. The claim is the graded answer; the reply's last
"Answer:" line or \\boxed{} is reported beside it, as the format wants the two to agree.
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics

HF_HOME = os.environ.get("HF_HOME", f"/scratch/{os.environ['USER']}/huggingface")
PARQUET = f"{HF_HOME}/hub/datasets--BytedTsinghua-SIA--AIME-2024/snapshots/*/data/aime-2024.parquet"
INTEGER = re.compile(r"-?\d[\d,]*")
FINAL = re.compile(r"Answer:\s*(.*)|\\boxed\{([^{}]*)\}")


def problems():
    """The upload repeats each of its 30 problems 32 times; returns them once, in order."""
    import pyarrow.parquet as parquet

    rows = parquet.read_table(glob.glob(PARQUET)[0]).to_pylist()
    distinct = {row["prompt"][0]["content"]: row["reward_model"]["ground_truth"] for row in rows}
    return list(distinct.items())


def integer(text):
    numbers = INTEGER.findall(text or "")
    return int(numbers[-1].replace(",", "")) if numbers else None


def reply_answer(reply):
    finals = [first or second for first, second in FINAL.findall(reply)]
    return integer(finals[-1]) if finals else None


def prompts(args):
    for index, (text, answer) in enumerate(problems()):
        for sample in range(args.repeat):
            print(json.dumps({"id": f"aime{index:02d}-{sample}", "user": text, "answer": answer}, ensure_ascii=False))


def grade(args):
    answers = [int(answer) for _, answer in problems()]
    totals = collections.Counter()
    per_problem = collections.Counter()
    lengths = []
    for line in open(args.results):
        result = json.loads(line)
        burst = result["bursts"][-1]
        problem = int(result["id"][4:6])
        totals["samples"] += 1
        lengths.append(burst["generated_tokens"])
        if burst["output"] is None:
            totals["truncated" if burst["finish_reason"] == "length" else "malformed"] += 1
            continue
        items = json.loads(burst["output"])["items"]
        claims = [item["payload"] for item in items if item["type"] == "claim"]
        reply = "\n".join(item["payload"] for item in items if item["type"] == "reply")
        claim, final = integer(claims[-1]) if claims else None, reply_answer(reply)
        totals["with_claim"] += bool(claims)
        totals["claim_correct"] += claim == answers[problem]
        totals["reply_correct"] += final == answers[problem]
        totals["claim_reply_differ"] += bool(claims) and final is not None and claim != final
        per_problem[problem] += claim == answers[problem]
    samples = totals["samples"]
    print(f"{args.results}\n{samples} samples of {len(answers)} problems")
    print(f"claim accuracy  {totals['claim_correct'] / samples:.3f} ({totals['claim_correct']}/{samples})")
    print(f"reply accuracy  {totals['reply_correct'] / samples:.3f} ({totals['reply_correct']}/{samples})")
    print(f"wrote a claim   {totals['with_claim']}, claim and reply differ {totals['claim_reply_differ']}")
    print(f"truncated {totals['truncated']}, malformed {totals['malformed']}")
    print(f"generated tokens: median {int(statistics.median(lengths))}, max {max(lengths)}")
    print(f"problems solved at least once: {len(+per_problem)}/{len(answers)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)
    make = commands.add_parser("prompts", help="print prompts for chat.py --prompts")
    make.add_argument("--repeat", type=int, default=4, help="samples per problem")
    make.set_defaults(run=prompts)
    check = commands.add_parser("grade", help="grade a results file from chat.py")
    check.add_argument("results")
    check.set_defaults(run=grade)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
