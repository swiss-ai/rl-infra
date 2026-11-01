"""Ask a running judge about a good and a bad answer, with the real prompts (see check_judge.sh).

It scores both with the rubrics, then has it pick between them with AlpacaEval's prompt in both orders.

Usage: check_judge.py <model> <chat completions URL>
"""

import json
import sys
import urllib.request
from pathlib import Path

import yaml
from resources_servers.rubric_judge import scoring

QUESTION = "Why is the sky blue during the day?"
GOOD = (
    "Sunlight contains all colours, and air molecules scatter short wavelengths far more strongly than "
    "long ones, an effect called Rayleigh scattering. Blue light is therefore scattered across the "
    "whole sky and reaches your eyes from every direction, while red light mostly passes straight through."
)
BAD = "The sky is blue because the sea reflects onto it. banana banana banana"
RUBRICS = ("helpfulness", "instruction_following", "truthfulness", "honesty")


def rubrics_path():
    import resources_servers.rubric_judge as package

    return Path(next(iter(package.__path__))) / "prompts" / "rubrics.yaml"


def top_logprobs(url, model, system, prompt):
    request = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "logprobs": True,
        "top_logprobs": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    post = urllib.request.Request(url, json.dumps(request).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(post, timeout=120) as response:
        first_token = json.load(response)["choices"][0]["logprobs"]["content"][0]
    return [(top["token"], top["logprob"]) for top in first_token["top_logprobs"]]


def main():
    model, url = sys.argv[1:3]
    rubric_file = yaml.safe_load(rubrics_path().read_text())
    failed = []
    for name in RUBRICS:
        scores = {}
        for label, answer in (("good", GOOD), ("bad", BAD)):
            prompt = rubric_file["rubrics"][name].format(prompt=QUESTION, completion=answer)
            top = top_logprobs(url, model, rubric_file["system"], prompt)
            score = scoring.expected_score(top)
            print(f"{name} {label}: expected verdict {score}, top tokens {top[:4]}", flush=True)
            scores[label] = score
        if scores["good"] is None or scores["bad"] is None or scores["good"] <= scores["bad"]:
            failed.append(name)
    alpaca = yaml.safe_load((rubrics_path().parent / "alpaca_eval.yaml").read_text())
    wins = []
    orders = ((GOOD, BAD, scoring.FIRST, scoring.SECOND), (BAD, GOOD, scoring.SECOND, scoring.FIRST))
    for output_1, output_2, winner, loser in orders:
        prompt = scoring.fill_pairwise(alpaca["user"], QUESTION, output_1, output_2)
        top = top_logprobs(url, model, alpaca["system"], prompt)
        wins.append(scoring.win_probability(top, winner, loser))
        print(f"alpaca_eval: win of the good answer {wins[-1]}, top tokens {top[:4]}", flush=True)
    if None in wins or sum(wins) / 2 <= 0.5:
        failed.append("alpaca_eval")
    if failed:
        sys.exit(f"ERROR: the judge did not prefer the good answer on: {', '.join(failed)}")
    print("== judge OK")


if __name__ == "__main__":
    main()
