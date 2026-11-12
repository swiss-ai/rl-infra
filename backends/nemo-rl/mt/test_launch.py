"""Exercise the real allocation launcher without Slurm, Ray, or a container."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class AllocationLaunchTests(unittest.TestCase):
    def test_existing_allocation_executes_ray_script_and_propagates_failure(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            clone = tmp / "nemo"
            clone.mkdir()
            (clone / "ray.sub").write_text(
                '#!/bin/bash\n# RAY_SINGLE_SRUN\n'
                'printf "%s\\n" "$SLURM_JOB_ID" "$SLURM_SUBMIT_DIR" "$COMMAND" > "$TEST_REPORT"\n'
                'exit 7\n')
            image = tmp / "image.sqsh"
            image.write_text("fixture")
            bins = tmp / "bin"
            bins.mkdir()
            sbatch = bins / "sbatch"
            sbatch.write_text('#!/bin/bash\ntouch "$TEST_SBATCH_CALLED"\nexit 99\n')
            sbatch.chmod(0o755)
            report = tmp / "report"
            env = {**os.environ, "SITE": "alps", "SCRATCH": str(tmp),
                   "NEMO_RL_DIR": str(clone), "IMAGE_WORK": str(image),
                   "OUT": str(tmp / "out"), "PREFLIGHT": "0", "DRY_RUN": "0",
                   "WANDB_PROJECT": "", "RUN_IN_ALLOCATION": "1",
                   "SLURM_JOB_ID": "777", "SLURM_JOB_NUM_NODES": "1",
                   "TEST_REPORT": str(report), "TEST_SBATCH_CALLED": str(tmp / "nested"),
                   "PATH": str(bins) + os.pathsep + os.environ["PATH"]}
            result = subprocess.run([
                "bash", str(root / "backends/nemo-rl/launch.sh"),
                str(root / "experiments/grpo-mt/qwen3-8b-smoke.yaml"), "1", "mt-test"
            ], env=env, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
            self.assertFalse((tmp / "nested").exists())
            lines = report.read_text().splitlines()
            self.assertEqual(lines[0], "777")
            self.assertEqual(lines[1], str(tmp / "out/slurm"))
            self.assertIn("cluster.num_nodes=1", lines[2])


if __name__ == "__main__":
    unittest.main()
