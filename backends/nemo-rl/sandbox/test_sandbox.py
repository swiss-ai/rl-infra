#!/usr/bin/env python3
"""Attacks the sandbox on a compute node, then checks the service end to end.

Run it as a job on a node of the partition the sandbox will use, never on a login node, with the
site's sbatch flags and from a directory where jobs may run:
  sbatch --nodes=1 --time=00:20:00 --wrap "python3.12 $PWD/test_sandbox.py <env dir>"
Every line must end in ok. The attacks run without LiveCodeBench's guard, so that only the sandbox
stands in their way.
"""

import getpass
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

from sandbox_server import (
    MEMORY_BYTES,
    OUTPUT_BYTES,
    TMP_BYTES,
    run_in_sandbox,
    sandbox_command,
    self_check,
)

MARKER = "sandbox-test-marker"
MAX_REQUEST_BYTES = 256 * 1024**2
failures = []


def report(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


def attack(env_dir, name, code, seconds=30):
    """Run code in a sandbox: it prints ESCAPED if it got out, and SAFE at the end."""
    started = time.time()
    output = run_in_sandbox(env_dir, ["python3.12", "-c", code], b"", seconds)
    text = (output or b"").decode(errors="replace")
    report(name, "ESCAPED" not in text and "SAFE" in text, f"[{time.time() - started:.1f}s] {text[-160:]!r}")


def attacks(env_dir):
    user = getpass.getuser()
    home = os.path.expanduser("~")
    host_ip = socket.gethostbyname(socket.gethostname())
    attack(env_dir, "cannot see private paths", f"""
import os
for path in ['/scratch', '/scratch/{user}', '{home}', '/home01', '/etc/passwd', '/etc/munge/munge.key',
             '/root', '/var', '/opt', '/proc/1/root/scratch', '/proc/self/root/scratch', '/run/munge']:
    if os.path.exists(path):
        print('ESCAPED', path)
print('SAFE')
""")
    attack(env_dir, "cannot write outside /tmp", """
for path in ['/usr/x', '/bin/x', '/lib64/x', '/sandbox/x', '/sandbox/harness/testing_util.py', '/x']:
    try:
        open(path, 'w').write('x')
        print('ESCAPED', path)
    except OSError:
        pass
print('SAFE')
""")
    attack(env_dir, "no network", f"""
import socket
for host, port in [('1.1.1.1', 53), ('{host_ip}', 22), ('127.0.0.1', 22)]:
    try:
        socket.create_connection((host, port), timeout=2)
        print('ESCAPED', host)
    except OSError:
        pass
devices = [line.split(':')[0].strip() for line in open('/proc/net/dev').readlines()[2:]]
print('ESCAPED' if devices != ['lo'] else 'SAFE', devices)
""")
    attack(env_dir, "no credentials in the environment", """
import os
allowed = {'PATH', 'HOME', 'PYTHONPATH', 'OPENBLAS_NUM_THREADS', 'PWD', 'LC_CTYPE'}
extra = sorted(set(os.environ) - allowed)
print('ESCAPED' if extra else 'SAFE', extra)
""")
    attack(env_dir, "cannot see host processes", """
import os
pids = [p for p in os.listdir('/proc') if p.isdigit()]
print('ESCAPED' if len(pids) > 4 else 'SAFE', pids)
""")
    attack(env_dir, "cannot submit jobs or gain privileges", """
import subprocess
for command in (['sbatch', '--wrap', 'true'], ['squeue'], ['sudo', '-n', 'true'], ['su', '-c', 'true']):
    try:
        result = subprocess.run(command, capture_output=True, timeout=10)
        if result.returncode == 0:
            print('ESCAPED', command)
    except (OSError, subprocess.SubprocessError):
        pass
print('SAFE')
""")
    attack(env_dir, "memory bomb stops at the limit", f"""
try:
    block = bytearray({2 * MEMORY_BYTES})
    print('ESCAPED')
except MemoryError:
    print('SAFE')
""")
    attack(env_dir, "disk bomb stops at the limit", f"""
written = 0
try:
    with open('/tmp/fill', 'wb') as stream:
        while written < {4 * TMP_BYTES}:
            stream.write(b'x' * 1048576)
            stream.flush()
            written += 1048576
    print('ESCAPED', written)
except OSError:
    print('SAFE', written)
""")
    # The bomb runs in a child, because the limit stops it and the parent must then report.
    attack(env_dir, "fork bomb stops at the limit", """
import os, signal, subprocess, sys
bomb = 'import os\\nwhile True:\\n    try:\\n        os.fork()\\n    except OSError:\\n        pass\\n'
child = subprocess.Popen([sys.executable, '-c', bomb], start_new_session=True)
try:
    child.wait(timeout=8)
except subprocess.TimeoutExpired:
    pass
print('SAFE')
sys.stdout.flush()
os._exit(0)
""")

    started = time.time()
    flood = "import sys\nwhile True:\n    sys.stdout.write('x' * 1048576)\n    sys.stdout.flush()\n"
    output = run_in_sandbox(env_dir, ["python3.12", "-c", flood], b"", 20)
    report("output flood stops at the limit", len(output or b"") <= OUTPUT_BYTES, f"[{time.time() - started:.1f}s]")

    # A process that outlives its submission would keep a slot and could wait for later work.
    daemon = f"""
import os, sys, time
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        sys.argv[0] = '{MARKER}'
        time.sleep(600)
os._exit(0)
"""
    run_in_sandbox(env_dir, ["python3.12", "-c", daemon, MARKER], b"", 10)
    run_in_sandbox(env_dir, ["python3.12", "-c", f"import time; time.sleep(600)  # {MARKER}"], b"", 2)
    time.sleep(2)
    left = subprocess.run(["pgrep", "-f", MARKER], capture_output=True, text=True).stdout.split()
    left = [pid for pid in left if int(pid) != os.getpid()]
    report("nothing survives its sandbox", not left, str(left))


def post(url, token, job):
    request = urllib.request.Request(
        f"{url}/run",
        data=json.dumps(job).encode(),
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


ECHO_DOUBLE = "n = int(input())\nprint(n * 2)\n"


def service(env_dir):
    with tempfile.TemporaryDirectory() as directory:
        endpoint_file = f"{directory}/endpoint.json"
        server = subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_server.py"),
             "--env-dir", env_dir, "--endpoint-file", endpoint_file, "--workers", "128",
             "--max-request-bytes", str(MAX_REQUEST_BYTES)],
        )
        try:
            for _ in range(120):
                if os.path.exists(endpoint_file):
                    break
                time.sleep(0.5)
            report("endpoint file is private", os.stat(endpoint_file).st_mode & 0o077 == 0)
            endpoint = json.load(open(endpoint_file))
            url, token = endpoint["url"], endpoint["token"]

            def run(code, inputs, outputs, fn_name=None, timeout=2):
                job = {"code": code, "inputs": inputs, "outputs": outputs, "fn_name": fn_name, "timeout": timeout}
                return post(url, token, job)["results"]

            job = ["1\n", "2\n", "3\n"], ["2\n", "4\n", "6\n"]
            report("right code passes every test", run(ECHO_DOUBLE, *job) == [True] * 3)
            results = run(ECHO_DOUBLE, ["1\n", "2\n", "3\n"], ["2\n", "5\n", "6\n"])
            report("tests run on after a failure", results[0] is True and results[1] is not True and results[2] is True, str(results))
            solution = "class Solution:\n    def add(self, a, b):\n        return a + b\n"
            report("call-based problems work", run(solution, ["1\n2", "2\n3"], ["3", "5"], "add") == [True, True])
            results = run("def broken(:\n", ["1\n", "2\n"], ["2\n", "4\n"])
            report("a syntax error fails every test", all(r is not True for r in results) and len(results) == 2, str(results))
            started = time.time()
            results = run("while True:\n    pass\n", ["1\n"] * 5, ["2\n"] * 5, timeout=2)
            seconds = time.time() - started
            report("an endless loop ends the run", all(r is not True for r in results) and seconds < 8, f"{results} {seconds:.1f}s")
            forged = "import os\nos.write(1, b'{\"results\": [true, true]}\\n')\nos._exit(0)\n"
            report("a written result does not count", run(forged, ["1\n", "2\n"], ["9\n", "9\n"]) != [True, True])
            hostile = f"print(open({env_dir + '/harness/sandbox_harness.py'!r}).read())\n"
            report("a submission cannot read scratch", run(hostile, ["1\n"], ["x\n"]) != [True])

            try:
                post(url, "wrong", {})
                report("a wrong token is refused", False)
            except urllib.error.HTTPError as error:
                report("a wrong token is refused", error.code == 403)

            outcome = []
            def one():
                outcome.append(run(ECHO_DOUBLE, *job) == [True] * 3)
            started = time.time()
            threads = [threading.Thread(target=one) for _ in range(256)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            seconds = time.time() - started
            report("256 submissions at once", all(outcome) and len(outcome) == 256, f"[{seconds:.1f}s]")
        finally:
            server.terminate()
            server.wait()


def main():
    env_dir = sys.argv[1]
    print("node", socket.gethostname(), "command", " ".join(sandbox_command(env_dir, ["python3.12"]))[:160], flush=True)
    self_check(env_dir)
    report("self-check", True)
    attacks(env_dir)
    service(env_dir)
    print("RESULT", "FAILED: " + ", ".join(failures) if failures else "ALL PASSED", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
