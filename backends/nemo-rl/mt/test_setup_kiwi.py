"""Ensure COMET setup cannot inherit the container's training constraints."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class KiwiSetupTests(unittest.TestCase):
    def test_install_is_outside_project_without_parent_constraints(self):
        script = Path(__file__).with_name("setup-kiwi.sh")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bins, project, venv = root / "bin", root / "project", root / "kiwi"
            bins.mkdir()
            project.mkdir()
            (project / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.13"\n')
            (venv / "bin").mkdir(parents=True)
            interpreter = venv / "bin/python"
            interpreter.write_text('#!/bin/sh\ncat >/dev/null\n')
            interpreter.chmod(0o755)
            uv = bins / "uv"
            uv.write_text(
                '#!/usr/bin/env python3\nimport os,sys,json\n'
                'with open(os.environ["KIWI_TEST_REPORT"], "a") as f:\n'
                ' f.write(json.dumps({"args":sys.argv[1:], "cwd":os.getcwd(), '
                '"constraints":{k:os.environ[k] for k in '
                '["UV_CONSTRAINT","PIP_CONSTRAINT","UV_BUILD_CONSTRAINT",'
                '"UV_OVERRIDE","UV_CONFIG_FILE","UV_PROJECT_ENVIRONMENT",'
                '"VIRTUAL_ENV","PYTHONPATH"] if k in os.environ}})+"\\n")\n')
            uv.chmod(0o755)
            report = root / "report.jsonl"
            env = {**os.environ, "PATH": str(bins) + os.pathsep + os.environ["PATH"],
                   "KIWI_VENV": str(venv), "KIWI_TEST_REPORT": str(report),
                   "UV_CONSTRAINT": "/training/constraints.txt", "PIP_CONSTRAINT": "/other",
                   "UV_BUILD_CONSTRAINT": "/build", "UV_OVERRIDE": "/overrides",
                   "UV_CONFIG_FILE": "/training/uv.toml", "UV_PROJECT_ENVIRONMENT": "/training",
                   "VIRTUAL_ENV": "/training", "PYTHONPATH": "/training"}
            result = subprocess.run(["bash", str(script)], cwd=project, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in report.read_text().splitlines()]
            self.assertEqual(len(calls), 2)
            for call in calls:
                self.assertEqual(call["constraints"], {})
                self.assertEqual(call["args"][0], "--no-config")
                self.assertNotEqual(call["cwd"], str(project))
                self.assertFalse(Path(call["cwd"]).exists())
            self.assertIn(str(venv / "bin/python"), calls[1]["args"])
