"""Pass-rate report for Enigmata from passrate_score.py's scores.jsonl.

  python3 -I analyze_passrate.py <scores.jsonl> <sample.jsonl> <Enigmata checkout> <out dir> [max_total_tokens]

Per answer, the frames are read back from the text (<|out|>NAME<|hdr|> ... <|/out|>), and the task's
verifier grades three texts: the claim, the reply, and the claim made to fit the task's extraction
(fenced in ``` when it has no code fence; wrapped in <begin_board>/<end_board> for campsite and
star_battle). The environment's reward (claim, else reply) is recomputed and compared with the
logged one.

Writes <out>/passrate.json (per task and difficulty), <out>/passrate.md (table) and
<out>/failures.md (wrong, complete answers per task to read for verifier false negatives).
"""

import collections
import importlib
import json
import os
import re
import signal
import sys

FRAME = re.compile(r"<\|out\|>(\w+)<\|hdr\|>(.*?)<\|/out\|>", re.S)
BOARD_TASKS = {"campsite", "star_battle"}
TABLE_TASKS = {"zebra_logic"}


class Timeout(BaseException):
    pass


def _raise(signum, frame):
    raise Timeout()


def frames(text):
    found = FRAME.findall(text)
    claims = [p for name, p in found if name == "claim"]
    replies = [p for name, p in found if name == "reply"]
    return (claims[-1] if claims else None), ("".join(replies) if replies else None)


def fitted(task, claim):
    if task in TABLE_TASKS:
        return claim
    if task in BOARD_TASKS:
        return claim if "<begin_board>" in claim else f"<begin_board>\n{claim.strip()}\n<end_board>"
    return claim if "```" in claim else f"```\n{claim.strip()}\n```"


def main():
    scores_path, sample_path, repo, out_dir = sys.argv[1:5]
    max_total = int(sys.argv[5]) if len(sys.argv) > 5 else 16384
    sys.path.insert(0, repo)
    os.makedirs(out_dir, exist_ok=True)
    with open(sample_path) as f:
        sample = [json.loads(l) for l in f]
    verifiers = {}
    signal.signal(signal.SIGALRM, _raise)

    def verify(task, text, spec):
        if not text or not text.strip():
            return 0.0, "empty"
        if task not in verifiers:
            verifiers[task] = importlib.import_module(f"verifiable_tasks.tasks.{task}.verifier").verify
        spec = json.loads(spec)  # fresh objects per call
        signal.setitimer(signal.ITIMER_REAL, 10.0)
        try:
            return float(verifiers[task](text, spec["answer"], spec["meta"])), "ok"
        except Timeout:
            return 0.0, "timeout"
        except Exception as e:
            return 0.0, f"error:{type(e).__name__}"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    cells = collections.defaultdict(list)
    failures = collections.defaultdict(list)
    mismatch = 0
    statuses = collections.Counter()
    with open(scores_path) as f:
        for line in f:
            r = json.loads(line)
            row = sample[r["idx"]]
            task, diff = r["task"], r["difficulty"]
            text = r["text"]
            claim, reply = frames(text)
            cut = (r["prompt_tokens"] + r["gen_tokens"] >= max_total - 1) or not text.rstrip().endswith("<|wait|>")
            c_score, c_st = verify(task, claim, row["output"])
            r_score, r_st = verify(task, reply, row["output"])
            f_score, _ = verify(task, fitted(task, claim), row["output"]) if claim and claim.strip() else (0.0, "empty")
            statuses.update([c_st, r_st])
            env_like = c_score if (claim and claim.strip()) else r_score
            if cut:
                env_like = 0.0 if r["reward"] == 0 else env_like
            if abs(env_like - r["reward"]) > 1e-6:
                mismatch += 1
            a = {
                "idx": r["idx"],
                "reward": r["reward"],
                "cut": cut,
                "has_claim": bool(claim and claim.strip()),
                "claim": c_score,
                "reply": r_score,
                "fitted_claim": f_score,
                "any": max(c_score, r_score, f_score),
                "tokens": r["gen_tokens"],
            }
            cells[(task, diff)].append(a)
            if r["reward"] == 0 and not cut and len(failures[task]) < 4:
                failures[task].append(
                    {
                        "difficulty": diff,
                        "claim": (claim or "")[:700],
                        "reply_tail": (reply or "")[-900:],
                        "gold_answer": json.dumps(json.loads(row["output"])["answer"])[:500],
                        "reply_score": r_score,
                        "fitted_claim_score": f_score,
                    }
                )

    report = {}
    for (task, diff), answers in sorted(cells.items()):
        n = len(answers)
        by_prompt = collections.defaultdict(list)
        for a in answers:
            by_prompt[a["idx"]].append(a["reward"] >= 0.5)
        mixed = sum(1 for v in by_prompt.values() if 0 < sum(v) < len(v))
        all_ok = sum(1 for v in by_prompt.values() if all(v))
        none_ok = sum(1 for v in by_prompt.values() if not any(v))
        complete = [a for a in answers if not a["cut"]]
        report[f"{task}/{diff}"] = {
            "answers": n,
            "puzzles": len(by_prompt),
            "reward": sum(a["reward"] for a in answers) / n,
            "puzzles_mixed": mixed,
            "puzzles_all_solved": all_ok,
            "puzzles_none_solved": none_ok,
            "cut_off": sum(a["cut"] for a in answers) / n,
            "no_claim_of_complete": (sum(not a["has_claim"] for a in complete) / len(complete)) if complete else None,
            "claim_score": sum(a["claim"] for a in answers) / n,
            "reply_score": sum(a["reply"] for a in answers) / n,
            "fitted_claim_score": sum(a["fitted_claim"] for a in answers) / n,
            "fallback_gain": sum(1 for a in answers if not a["has_claim"] and a["reply"] >= 0.5) / n,
            "fit_gain": sum(1 for a in answers if a["has_claim"] and a["claim"] < 0.5 and a["fitted_claim"] >= 0.5) / n,
            "reply_right_claim_wrong": sum(1 for a in answers if a["has_claim"] and a["claim"] < 0.5 and a["reply"] >= 0.5) / n,
            "mean_tokens": sum(a["tokens"] for a in answers) / n,
        }
    total = sum(len(v) for v in cells.values())
    summary = {
        "answers": total,
        "reward": sum(a["reward"] for v in cells.values() for a in v) / total,
        "cut_off": sum(a["cut"] for v in cells.values() for a in v) / total,
        "env_recompute_mismatches": mismatch,
        "verifier_statuses": dict(statuses),
    }
    with open(os.path.join(out_dir, "passrate.json"), "w") as f:
        json.dump({"summary": summary, "cells": report}, f, indent=1)

    cols = ["reward", "puzzles_mixed", "puzzles_all_solved", "puzzles_none_solved", "cut_off",
            "no_claim_of_complete", "claim_score", "reply_score", "fitted_claim_score", "fallback_gain",
            "fit_gain", "mean_tokens"]
    with open(os.path.join(out_dir, "passrate.md"), "w") as f:
        f.write("| cell | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n")
        for key, v in report.items():
            vals = []
            for c in cols:
                x = v[c]
                vals.append("-" if x is None else (f"{x:.0f}" if c == "mean_tokens" else
                                                   (str(x) if isinstance(x, int) else f"{x:.2f}")))
            f.write(f"| {key} | " + " | ".join(vals) + " |\n")
        f.write("\n" + json.dumps(summary) + "\n")
    with open(os.path.join(out_dir, "failures.md"), "w") as f:
        for task, items in sorted(failures.items()):
            f.write(f"## {task}\n\n")
            for it in items:
                f.write(f"### {it['difficulty']} (reply score {it['reply_score']}, fitted claim {it['fitted_claim_score']})\n")
                f.write(f"CLAIM:\n~~~~text\n{it['claim']}\n~~~~\nREPLY TAIL:\n~~~~text\n{it['reply_tail']}\n~~~~\n")
                f.write(f"GOLD: {it['gold_answer']}\n\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
