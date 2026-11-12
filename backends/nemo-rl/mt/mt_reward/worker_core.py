"""Scoring implementation used by the GPU Ray actor."""

from .scoring import Example, TranslationReward


class RewardWorker:
    def __init__(self, config):
        from .models import WalarComponents

        # Ray sets CUDA_VISIBLE_DEVICES to the allocated GPU. Preserve it and
        # always use logical device zero inside this actor.
        config = {**config, "device": "cuda:0"}
        self.scorer = TranslationReward(
            WalarComponents(config),
            alignment_weight=config.get("alignment_weight", 20),
            chrf_weight=config.get("chrf_weight", 20))

    def ready(self):
        return True

    def score(self, examples):
        return [self.scorer.score(Example(**row)).asdict() for row in examples]
