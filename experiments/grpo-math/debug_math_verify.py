"""Show how NeMo-RL's hf_math_verify reward parses responses and ground truths.

For every case: which regex (priority, name, parser.py line) matched, what text it
matched, what sympy value came out, and the score from the exact call NeMo-RL makes
(nemo_rl/environments/math_environment.py:68-74 and :107-110). Needs math-verify 0.9.0.

  python debug_math_verify.py                          # built-in cases
  python debug_math_verify.py --response '... \\boxed{3}' --gt '3'
  python debug_math_verify.py --show-regex             # print the full compiled regexes
"""

import argparse
import logging
from itertools import groupby

from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    extract_match,
    get_extraction_regexes,
)

logging.getLogger("math_verify").setLevel(logging.CRITICAL)

# Registration order in math_verify 0.9.0 parser.py (line where each is appended).
EXPR_NAMES = [
    "expr  'final answer is X. I hope'      (parser.py:220)",
    "expr  'final answer ... is X'          (parser.py:221)",
    "expr  'answer: ... X'                  (parser.py:230)",
    "expr  'answer ... X'                   (parser.py:230)",
    "expr  plain arithmetic expression      (parser.py:234)",
    "expr  plain number                     (parser.py:235)",
]
LATEX_NAMES = [
    "latex 'final answer is $X$. I hope'    (parser.py:327)",
    "latex 'final answer ... is $X$'        (parser.py:328)",
    "latex 'answer: ... $X$'                (parser.py:334)",
    "latex 'answer ... $X$'                 (parser.py:334)",
    "latex plain latex block                (parser.py:338)",
    "latex latex block containing \\boxed{} (parser.py:350)",
    "latex bare \\boxed{...}                (parser.py:353)",
]

PRED_CONFIGS = (ExprExtractionConfig(), LatexExtractionConfig())
GOLD_CONFIGS = (LatexExtractionConfig(),)

# Same object NeMo-RL builds in HFVerifyWorker.__init__.
verify_func = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)

CASES = [
    ("boxed integer", "So the total is \\boxed{5}.", "5"),
    ("boxed fraction vs decimal gt", "Thus \\boxed{\\frac{1}{2}}.", "0.5"),
    ("two boxes: rightmost wins", "First guess \\boxed{3}, but actually \\boxed{4}.", "4"),
    ("box, then 'final answer is' to its right", "\\boxed{3}. The final answer is 5", "3"),
    ("numbers after the box", "\\boxed{7}. Check: 3+4=7, done in 2 steps.", "7"),
    ("no box, 'Answer:'", "Answer: 42", "42"),
    ("no box, no anchor, last number is right", "12 apples plus 7 more gives 19", "19"),
    ("no box, no anchor, last number is wrong", "The result is 19 after 3 steps", "19"),
    ("inline latex only", "So $x = \\sqrt{2}$", "\\sqrt{2}"),
    ("thousand separator", "\\boxed{1,000}", "1000"),
    ("percent", "\\boxed{50\\%}", "0.5"),
    ("pi expression", "\\boxed{2\\pi}", "2\\pi"),
    ("algebraic equivalence", "\\boxed{(x+1)^2}", "x^2+2x+1"),
    ("equation vs number", "\\boxed{x=3}", "3"),
    ("interval vs inequality gt", "\\boxed{[-2,7]}", "-2 \\le x \\le 7"),
    ("two boxed answers", "\\boxed{2} and \\boxed{3}", "2,3"),
    ("truncated box (hit max length)", "so the answer is \\boxed{12", "12"),
    ("no answer at all", "I am not sure how to solve this.", "5"),
]


def named_patterns(configs):
    out = []
    for patterns, target_type in get_extraction_regexes(configs):
        names = EXPR_NAMES if isinstance(target_type, ExprExtractionConfig) else LATEX_NAMES
        assert len(patterns) == len(names), "math-verify version differs from 0.9.0"
        for (pattern, priority), name in zip(patterns, names):
            out.append((pattern, target_type, priority, name))
    return out


def trace(text, configs):
    """Mirror of parser.extract_target_from_pred (parser.py:573-646), printing each step."""
    patterns = sorted(named_patterns(configs), key=lambda x: x[2])
    for priority, group in groupby(patterns, key=lambda x: x[2]):
        matches = [
            (m, m.start(), m.end(), target_type, name)
            for pattern, target_type, _, name in group
            for m in pattern.finditer(text)
        ]
        matches.sort(key=lambda x: (x[2], -x[1]), reverse=True)
        for m, start, end, target_type, name in matches:
            extracted, _ = extract_match(m, target_type)
            status = f"-> {extracted!r}" if extracted is not None else "-> failed to parse, try next"
            print(f"    [prio {priority:>3}] {name}  matched {m.group(0)!r} at {start}-{end} {status}")
            if extracted is not None:
                return extracted
    print("    no regex matched")
    return None


def run_case(name, response, gt):
    gold_text = "\\boxed{" + gt + "}"
    print(f"=== {name}")
    print(f"  response: {response!r}")
    print(f"  gt: {gt!r}  -> passed to verifier as {gold_text!r}")
    print("  response parsing:")
    trace(response, PRED_CONFIGS)
    print("  gt parsing:")
    trace(gold_text, GOLD_CONFIGS)
    try:
        score, _ = verify_func([gold_text], [response])
    except (Exception, TimeoutException) as e:  # NeMo-RL gives 0.0 on any exception
        print(f"  exception {type(e).__name__}: {e}")
        score = 0.0
    print(f"  REWARD: {float(score)}\n")


def show_regex():
    for label, configs in (("RESPONSE", PRED_CONFIGS), ("GROUND TRUTH", GOLD_CONFIGS)):
        print(f"##### regexes applied to the {label}")
        for pattern, _, priority, name in sorted(named_patterns(configs), key=lambda x: x[2]):
            print(f"[prio {priority:>3}] {name}  ({len(pattern.pattern)} chars)\n{pattern.pattern}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--response")
    ap.add_argument("--gt")
    ap.add_argument("--show-regex", action="store_true")
    args = ap.parse_args()
    if args.show_regex:
        show_regex()
    elif args.response is not None:
        run_case("custom", args.response, args.gt)
    else:
        for case in CASES:
            run_case(*case)
