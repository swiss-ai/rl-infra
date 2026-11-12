"""Resource validation and Ray actor options shared by setup and tests."""

import os
import json
from pathlib import Path


def load_scorer_config(config):
    if "scorer" in config and "scorer_config" in config:
        raise ValueError("Specify either scorer or scorer_config, not both")
    if "scorer" in config:
        scorer = config["scorer"]
    else:
        with open(config["scorer_config"]) as stream:
            scorer = json.load(stream)
    if not isinstance(scorer, dict):
        raise ValueError("The scorer configuration must be a dictionary")
    return scorer


def worker_options(config, env_vars=None):
    resources = config["resources"]
    if resources["num_nodes"] != 1 or resources["gpus_per_node"] != 1:
        raise ValueError("The MT reward worker requires num_nodes=1, gpus_per_node=1")
    python = config["scorer_python"]
    if not Path(python).is_absolute():
        raise ValueError("scorer_python must be an absolute path visible on every node")
    variables = dict(os.environ if env_vars is None else env_vars)
    # The driver runs in NeMo-RL's venv; the scorer must use its own venv.
    for key in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME",
                "CUDA_VISIBLE_DEVICES", "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"):
        variables.pop(key, None)
    variables["VIRTUAL_ENV"] = str(Path(python).parent.parent)
    return {
        "num_gpus": 1,
        "num_cpus": resources.get("num_cpus", 4),
        "runtime_env": {"py_executable": python, "env_vars": variables},
    }
