"""NeMo-RL rollout conversion and metrics, without Ray or Torch imports."""

from collections import defaultdict
import json

from .scoring import Example


def examples_from_rollouts(message_log_batch, metadata):
    if len(message_log_batch) != len(metadata):
        raise ValueError("Message and metadata batch sizes differ")
    examples = []
    for messages, info in zip(message_log_batch, metadata):
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("MT reward requires a final assistant translation")
        examples.append(Example(**{key: info[key] for key in
            ("source_lang", "target_lang", "source", "reference")},
            hypothesis=messages[-1]["content"]))
    return examples


def rollout_metrics(batch):
    groups = defaultdict(list)
    for messages, info in zip(batch["message_log"], batch["extra_env_info"]):
        for message in reversed(messages):
            if message["role"] == "environment":
                score = json.loads(message["content"])["mt_reward"]
                groups["mt"].append(score)
                groups[f"mt/{info['source_lang']}-to-{info['target_lang']}"].append(score)
                break
    metrics = {}
    for prefix, rows in groups.items():
        metrics[f"{prefix}/num_samples"] = len(rows)
        for key in ("reward", "language_alignment", "metricx", "word_alignment", "chrf_pp"):
            values = [row[key] for row in rows if row[key] is not None]
            if values:
                metrics[f"{prefix}/{key}/mean"] = sum(values) / len(values)
    return metrics
