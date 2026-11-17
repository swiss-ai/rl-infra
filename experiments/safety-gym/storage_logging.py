"""Process-local rollout cadence; installed in driver and collector workers.

No NeMo-RL source files are edited. Metrics remain enabled every step.
"""

import os
from pathlib import Path
import re
import sys


def keep_dump(filename, period):
    """Retain training dumps at optimizer step multiples, skip validation dumps."""
    match = re.fullmatch(r"train_weights(\d+)\.jsonl", Path(filename).name)
    if match:
        # Async weights N produce optimizer step N+1. The stock synchronous
        # path is unsupported by this async-only experiment adapter.
        return (int(match.group(1)) + 1) % period == 0
    return False


def install():
    """Install a bounded dump-path selector in this Python process."""
    import nemo_rl.experience.rollouts as rollouts

    if getattr(rollouts, "_safety_logging_installed", False):
        return
    period = int(os.environ["SAFETY_ROLLOUT_SAVE_PERIOD"])
    if period <= 0:
        raise ValueError("SAFETY_ROLLOUT_SAVE_PERIOD must be positive")
    original = rollouts.nemo_gym_rollout_dump_path

    def selected_path(env_config, filename):
        return original(env_config, filename) if keep_dump(filename, period) else None

    rollouts.nemo_gym_rollout_dump_path = selected_path
    # GRPO imports this symbol by value, while the collector imports it inside
    # its rollout method. Cover both import forms.
    grpo = sys.modules.get("nemo_rl.algorithms.grpo")
    if grpo is not None:
        grpo.nemo_gym_rollout_dump_path = selected_path
    rollouts._safety_logging_installed = True


def install_driver():
    """Install driver policy and attach the same hook to the Ray collector."""
    import nemo_rl.algorithms.grpo as grpo

    install()
    original = grpo.make_actor_runtime_env

    def runtime_env(actor_name, *args, **kwargs):
        result = original(actor_name, *args, **kwargs)
        if actor_name.endswith(".AsyncTrajectoryCollector"):
            if result.get("worker_process_setup_hook"):
                raise RuntimeError("collector already has a worker setup hook")
            result["worker_process_setup_hook"] = "storage_logging.install"
            env = result.setdefault("env_vars", {})
            directory = str(Path(__file__).absolute().parent)
            env["PYTHONPATH"] = directory + os.pathsep + env.get("PYTHONPATH", "")
            env["SAFETY_ROLLOUT_SAVE_PERIOD"] = os.environ["SAFETY_ROLLOUT_SAVE_PERIOD"]
        return result

    grpo.make_actor_runtime_env = runtime_env
