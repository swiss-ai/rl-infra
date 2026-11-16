"""Persistent COMET-Kiwi subprocess, isolated from the Python 3.13 Ray scorer."""

import json
import os
from pathlib import Path
import subprocess

from .scoring import bounded


class CometKiwiReward:
    def __init__(self, config):
        python = config.get("cometkiwi_python")
        if not python or not Path(python).is_absolute():
            raise ValueError("Enabled COMET-Kiwi requires an absolute cometkiwi_python path")
        env = {key: value for key, value in os.environ.items()
               if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}}
        self.process = subprocess.Popen(
            [python, "-u", str(Path(__file__).with_name("kiwi_process.py")),
             "--model", config.get("cometkiwi_model", "Unbabel/wmt22-cometkiwi-da"),
             "--device", config.get("device", "cuda:0")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        try:
            if self._read() != {"ready": True}:
                raise RuntimeError("Unexpected COMET-Kiwi readiness response")
        except BaseException:
            self.close()
            raise

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("COMET-Kiwi process exited; inspect its stderr in the job log")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(f"COMET-Kiwi failed: {result['error']}")
        return result

    def score(self, source, hypothesis):
        self.process.stdin.write(json.dumps({"src": source, "mt": hypothesis}) + "\n")
        self.process.stdin.flush()
        return bounded(self._read()["score"], 0, 1, "COMET-Kiwi")

    def close(self):
        process = getattr(self, "process", None)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()

    def __del__(self):
        self.close()
