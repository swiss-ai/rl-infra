"""Register the MT environment before invoking the pinned upstream trainer."""

import runpy

from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.utils import register_env
from nemo_rl.data.processors import register_processor
from mt_reward.processor import translation_data_processor

ACTOR = "mt_reward.environment.TranslationEnvironment"
register_env("mt", ACTOR)
ACTOR_ENVIRONMENT_REGISTRY[ACTOR] = PY_EXECUTABLES.SYSTEM
register_processor("translation_data_processor", translation_data_processor)

if __name__ == "__main__":
    runpy.run_path("/opt/nemo-rl/examples/run_grpo.py", run_name="__main__")
