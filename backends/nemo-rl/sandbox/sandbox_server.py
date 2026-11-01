#!/usr/bin/env python3
"""Runs model-written code for NeMo Gym, on a node of its own.

Gym's code_gen server posts a submission and its unit tests here. Each submission runs under
bubblewrap with no view of the cluster's file systems, no network and no credentials, and with
limits on memory, processes, files and time. Standard library only, because this runs on the host.
"""

import argparse
import hmac
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MEMORY_BYTES = 4 * 1024**3
OUTPUT_BYTES = 8 * 1024**2
TMP_BYTES = 64 * 1024**2
MAX_PROCESSES = 1024
MAX_OPEN_FILES = 256
MAX_REQUEST_BYTES = 256 * 1024**2
MAX_SECONDS = 300
# Allows for start-up and for unit tests that end early.
GRACE_SECONDS = 5
FAILED_RESULT = -1
# Must be missing in a sandbox. The self-check refuses to serve if one of them is there.
PRIVATE_PATHS = ("/scratch", "/home01", "/etc/munge", "/etc/passwd", "/opt")


def sandbox_command(env_dir, argv):
    """Return the command that runs argv in a new sandbox.

    Args:
        env_dir: Directory of the Python packages and harness, mounted read-only at /sandbox.
        argv: The command to run inside.

    Returns:
        The full command line.
    """
    limits = [
        "prlimit",
        f"--as={MEMORY_BYTES}",
        f"--fsize={OUTPUT_BYTES}",
        f"--nproc={MAX_PROCESSES}",
        f"--nofile={MAX_OPEN_FILES}",
        "--core=0",
        "--",
    ]
    # Nothing is mounted unless it is listed here, so /scratch and home do not exist inside.
    system = []
    for path in ("/usr", "/lib", "/lib64", "/bin", "/etc/ld.so.cache"):
        system += ["--ro-bind", path, path]
    return [
        *limits,
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--hostname",
        "sandbox",
        *system,
        "--ro-bind",
        env_dir,
        "/sandbox",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--size",
        str(TMP_BYTES),
        "--tmpfs",
        "/tmp",
        "--remount-ro",
        "/",
        "--chdir",
        "/tmp",
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PYTHONPATH",
        "/sandbox/site:/sandbox/harness",
        "--setenv",
        "OPENBLAS_NUM_THREADS",
        "1",
        *argv,
    ]


def run_in_sandbox(env_dir, argv, stdin_bytes, seconds):
    """Run a command in a sandbox and return what it printed.

    Args:
        env_dir: See sandbox_command.
        argv: The command to run inside.
        stdin_bytes: What the command reads.
        seconds: Time limit. The sandbox and everything in it is killed at the limit.

    Returns:
        Its output, or None if it hit the time limit.
    """
    # Files, not pipes: the file size limit then bounds output that a submission writes itself.
    with tempfile.TemporaryFile() as stdin_file, tempfile.TemporaryFile() as stdout_file:
        stdin_file.write(stdin_bytes)
        stdin_file.seek(0)
        process = subprocess.Popen(
            sandbox_command(env_dir, argv),
            stdin=stdin_file,
            stdout=stdout_file,
            stderr=subprocess.DEVNULL,
        )
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return None
        stdout_file.seek(0)
        return stdout_file.read(OUTPUT_BYTES)


def run_unit_tests(env_dir, job):
    """Run one submission's unit tests.

    Args:
        env_dir: See sandbox_command.
        job: The code, inputs, outputs, fn_name and per-test timeout.

    Returns:
        One result per test, True where it passed, and details of the first failure. A sandbox
        that hit the time limit or printed something else fails every test.
    """
    num_tests = len(job["inputs"])
    seconds = min((job["timeout"] + 1) * num_tests + GRACE_SECONDS, MAX_SECONDS)
    output = run_in_sandbox(
        env_dir,
        ["python3.12", "/sandbox/harness/sandbox_harness.py"],
        json.dumps(job).encode(),
        seconds,
    )
    failed = {"results": [FAILED_RESULT] * num_tests, "metadata": {"error": "no result"}}
    if output is None:
        return {**failed, "metadata": {"error": "sandbox time limit"}}
    try:
        report = json.loads(output.splitlines()[-1])
        results = report["results"]
    except (IndexError, ValueError, KeyError, TypeError):
        return failed
    if not isinstance(results, list) or len(results) != num_tests:
        return failed
    return {"results": results, "metadata": report.get("metadata")}


def self_check(env_dir):
    """Refuse to serve unless a sandbox really is cut off from this node.

    Args:
        env_dir: See sandbox_command.

    Raises:
        RuntimeError: Something that must be out of reach can be reached.
    """
    probe = (
        "import json, os, socket\n"
        f"paths = [p for p in {PRIVATE_PATHS!r} if os.path.exists(p)]\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    network = True\n"
        "except OSError:\n"
        "    network = False\n"
        "print(json.dumps({'paths': paths, 'network': network, 'env': sorted(os.environ)}))\n"
    )
    output = run_in_sandbox(env_dir, ["python3.12", "-c", probe], b"", 30)
    if not output:
        raise RuntimeError("The sandbox did not start: is bubblewrap allowed on this node?")
    report = json.loads(output)
    if report["paths"] or report["network"]:
        raise RuntimeError(f"The sandbox is not isolated: {report}")


class Server(ThreadingHTTPServer):
    """Takes a whole batch of rollouts at once."""

    request_queue_size = 1024
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    """Serves POST /run for the holder of the token."""

    token = ""
    env_dir = ""
    slots = threading.BoundedSemaphore(1)

    def log_message(self, format, *args):
        pass

    def _reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        token = self.headers.get("Authorization", "")
        if not hmac.compare_digest(token, f"Bearer {self.token}"):
            self._reply(403, {"error": "bad token"})
            return
        length = int(self.headers.get("Content-Length", 0))
        if self.path != "/run" or length > MAX_REQUEST_BYTES:
            self._reply(400, {"error": "expected POST /run within the size limit"})
            return
        job = json.loads(self.rfile.read(length))
        with self.slots:
            self._reply(200, run_unit_tests(self.env_dir, job))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-dir", required=True)
    parser.add_argument("--endpoint-file", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()

    self_check(args.env_dir)
    Handler.token = secrets.token_urlsafe(32)
    Handler.env_dir = args.env_dir
    Handler.slots = threading.BoundedSemaphore(args.workers)
    server = Server(("0.0.0.0", 0), Handler)
    host = socket.gethostbyname(socket.gethostname())
    endpoint = {"url": f"http://{host}:{server.server_address[1]}", "token": Handler.token}

    # Only this user may read the token: the file is created private, then moved into place.
    temporary = f"{args.endpoint_file}.{os.getpid()}"
    with open(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(endpoint, stream)
    os.replace(temporary, args.endpoint_file)
    print(f"Serving {endpoint['url']} with {args.workers} workers", flush=True)
    # scancel sends SIGTERM: the endpoint file must not outlive the server.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    try:
        server.serve_forever()
    finally:
        os.unlink(args.endpoint_file)


if __name__ == "__main__":
    main()
