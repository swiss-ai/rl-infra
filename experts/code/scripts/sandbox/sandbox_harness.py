"""Runs one submission's unit tests inside a sandbox: job on stdin, one line of JSON on stdout.

Each test runs on its own, because LiveCodeBench's runner stops at the first failure and so cannot
tell how many tests a submission passes. Each runs in a process of its own too, because the runner
disables parts of the os module for good, and a crash then costs only that test. A timeout ends the
run: every later test would likely time out as well, and spend the whole time limit.
"""

import json
import os
import sys

from testing_util import run_test

TIME_LIMIT_EXCEEDED = -3
RUNTIME_ERROR = -4
ERROR_BEFORE_TESTING = "Error during testing"


def run_one_test(job, test_input, test_output, report_fd):
    """Run one test in a child process.

    Args:
        job: The code, fn_name and per-test timeout.
        test_input: The test's input.
        test_output: The expected output.
        report_fd: File descriptor of the report, which the child must not hold.

    Returns:
        The runner's result and details.
    """
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        os.close(report_fd)
        in_outs = {"inputs": [test_input], "outputs": [test_output], "fn_name": job["fn_name"]}
        result, metadata = run_test(in_outs, test=job["code"], debug=False, timeout=job["timeout"])
        with os.fdopen(write_fd, "w") as stream:
            json.dump({"result": result, "metadata": metadata}, stream, default=str)
        os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd) as stream:
        text = stream.read()
    os.waitpid(pid, 0)
    try:
        outcome = json.loads(text)
        return outcome["result"], outcome["metadata"]
    except (ValueError, KeyError, TypeError):
        return [RUNTIME_ERROR], {"error_message": "The test process died."}


def main():
    sys.set_int_max_str_digits(50000)
    job = json.load(sys.stdin)
    # A submission that writes to file descriptor 1 must not reach or spoil the report.
    report_fd = os.dup(1)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    results = []
    first_failure = None
    for test_input, test_output in zip(job["inputs"], job["outputs"]):
        result, metadata = run_one_test(job, test_input, test_output, report_fd)
        passed = result == [True]
        results.append(True if passed else result[-1])
        if passed:
            continue
        first_failure = first_failure or metadata
        message = str((metadata or {}).get("error_message", ""))
        if result[-1] == TIME_LIMIT_EXCEEDED or message.startswith(ERROR_BEFORE_TESTING):
            results += [results[-1]] * (len(job["inputs"]) - len(results))
            break
    with os.fdopen(report_fd, "w") as report:
        report.write(json.dumps({"results": results, "metadata": first_failure}, default=str) + "\n")


if __name__ == "__main__":
    main()
