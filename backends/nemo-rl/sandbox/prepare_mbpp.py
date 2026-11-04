#!/usr/bin/env python3
"""Turns MBPP, a set of short Python tasks, into rows of Gym's code_gen server.

Each MBPP task has plain tests such as `assert add(1, 2) == 3`, which become call-based unit tests, so
code_gen and the sandbox need no change. Tasks 11-510 are MBPP's test split and become validation, the
rest become training. A task is left out when a test is not a plain equality on literals, or when an
argument is a tuple or a set: JSON would hand the function a list, and correct code could fail on it.
Prints what it left out.

    curl -LO https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl
    python3 prepare_mbpp.py --input mbpp.jsonl --output-dir $SCRATCH/code-gen-mbpp
"""

import argparse
import ast
import json
import os
from collections import Counter

AGENT_NAME = "code_gen_simple_agent"
TEST_SPLIT = range(11, 511)
PROMPT = (
    "You are an expert Python programmer, and here is your task: {text} Your code should pass these "
    "tests:\n\n{tests}\n\nPut your solution in a ```python code block."
)


def has_no_json_form(value, *, as_argument):
    """Whether JSON would change a value, in a way that could fail correct code."""
    if isinstance(value, (set, frozenset, bytes, complex)):
        return True
    if isinstance(value, tuple) and as_argument:
        return True
    if isinstance(value, (list, tuple)):
        return any(has_no_json_form(item, as_argument=as_argument) for item in value)
    if isinstance(value, dict):
        return any(
            not isinstance(key, str) or has_no_json_form(item, as_argument=as_argument)
            for key, item in value.items()
        )
    return False


def parse_test(test):
    """Split `assert name(args) == expected`.

    Args:
        test: One line of a task's test_list.

    Returns:
        The function's name, its arguments and the expected value, or a reason to leave the task out.
    """
    node = ast.parse(test.strip()).body[0]
    if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
        return "not an equality"
    comparison = node.test
    if len(comparison.ops) != 1 or not isinstance(comparison.ops[0], ast.Eq):
        return "not an equality"
    call = comparison.left
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.keywords:
        return "not a plain call"
    try:
        arguments = [ast.literal_eval(argument) for argument in call.args]
        expected = ast.literal_eval(comparison.comparators[0])
    except (ValueError, TypeError, SyntaxError):
        return "not literals"
    if has_no_json_form(arguments, as_argument=True):
        return "tuple or set argument"
    if has_no_json_form(expected, as_argument=False):
        return "expected value has no JSON form"
    return call.func.id, arguments, expected


def to_row(task):
    """Build a code_gen row.

    Args:
        task: One line of mbpp.jsonl.

    Returns:
        The row, or a reason to leave the task out.
    """
    if task["test_setup_code"]:
        return "needs setup code"
    names = set()
    inputs = []
    outputs = []
    for test in task["test_list"]:
        parsed = parse_test(test)
        if isinstance(parsed, str):
            return parsed
        name, arguments, expected = parsed
        names.add(name)
        inputs.append("\n".join(json.dumps(argument) for argument in arguments))
        outputs.append(json.dumps(expected))
    if len(names) != 1:
        return "tests call different functions"
    prompt = PROMPT.format(text=task["text"], tests="\n".join(task["test_list"]))
    return {
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
        "verifier_metadata": {
            "problem_id": str(task["task_id"]),
            "unit_tests": {"inputs": inputs, "outputs": outputs, "fn_name": names.pop()},
        },
        "agent_ref": {"type": "responses_api_agents", "name": AGENT_NAME},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = {"train": [], "validation": []}
    left_out = Counter()
    with open(args.input) as source:
        for line in source:
            task = json.loads(line)
            row = to_row(task)
            if isinstance(row, str):
                left_out[row] += 1
            else:
                rows["validation" if task["task_id"] in TEST_SPLIT else "train"].append(row)
    for name, split in rows.items():
        with open(os.path.join(args.output_dir, f"{name}.jsonl"), "w") as output:
            for row in split:
                output.write(json.dumps(row) + "\n")
        print(f"{name}: {len(split)} tasks")
    print(f"left out {sum(left_out.values())}: {dict(left_out)}")


if __name__ == "__main__":
    main()
