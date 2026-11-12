"""GPU reward actor, running in its own Python environment within training Ray."""

import ray

from .worker_core import RewardWorker


TranslationRewardWorker = ray.remote(RewardWorker)
