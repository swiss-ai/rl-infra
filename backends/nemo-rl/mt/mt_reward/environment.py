"""Single-turn NeMo-RL environment for WALAR + chrF++."""

import json

import ray
import torch

from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

from .rollouts import examples_from_rollouts, rollout_metrics
from .runtime import load_scorer_config, worker_options
from .worker import TranslationRewardWorker


@ray.remote
class TranslationEnvironment(EnvironmentInterface):
    def __init__(self, config):
        self.timeout = float(config.get("timeout", 600))
        self.worker = TranslationRewardWorker.options(**worker_options(config)).remote(
            load_scorer_config(config))
        # Wait for the GPU allocation and model loading before accepting rollouts.
        ray.get(self.worker.ready.remote(), timeout=self.timeout)

    def step(self, message_log_batch, metadata):
        examples = examples_from_rollouts(message_log_batch, metadata)
        scores = ray.get(self.worker.score.remote([e.__dict__ for e in examples]),
                         timeout=self.timeout) if examples else []
        # Store per-rollout scores in the terminal observation: single-turn
        # rollout collection does not retain returned metadata on termination.
        observations = [{"role": "environment", "content": json.dumps({"mt_reward": score})}
                        for score in scores]
        return EnvironmentReturn(
            observations=observations, metadata=metadata,
            next_stop_strings=[None] * len(examples),
            rewards=torch.tensor([s["reward"] for s in scores], dtype=torch.float32),
            terminateds=torch.ones(len(examples), dtype=torch.bool),
            answers=[e.hypothesis for e in examples])

    def global_post_process_and_metrics(self, batch):
        return batch, rollout_metrics(batch)

    def shutdown(self):
        ray.kill(self.worker)
