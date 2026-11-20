"""Enigmata pass rates of a checkpoint with vLLM only, no trainer, graded by the native enigmata
environment (a copy of rl-analysis/thinking-levels/think_score.py with task/difficulty per answer).

Run through run.sh like a GRPO run (SCRIPT=<this file>), so the config is the same recipe + task +
overrides; the prompts are the task's training set, built by GRPO's own setup_response_data (same
processor, Apertus 2 prompt format and environment). vLLM loads the checkpoint itself in BF16
(Apertus 2 rejects vLLM's online FP8). Each prompt is answered K times through
run_async_multi_turn_rollout, the rollout GRPO's validation uses, so each answer gets the
training reward.

Every finished batch is appended to <logger.log_dir>/scores.jsonl, one line per answer:
{"idx", "task", "difficulty", "reward", "prompt_tokens", "gen_tokens", "text"}; a resubmitted job
skips the prompts already scored. analyze_passrate.py reads it.

Task-file options:
  score:
    num_generations: 8      # answers per prompt
    batch_prompts: 216      # prompts per rollout batch
"""
import argparse
import json
import os
import pprint
import sys
import time

sys.path.insert(0, "/opt/nemo-rl")

from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from nemo_rl.algorithms.grpo import MasterConfig  # noqa: E402
from nemo_rl.algorithms.utils import get_tokenizer  # noqa: E402
from nemo_rl.data.collate_fn import rl_collate_fn  # noqa: E402
from nemo_rl.data.utils import setup_response_data  # noqa: E402
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray  # noqa: E402
from nemo_rl.environments.utils import shutdown_environments  # noqa: E402
from nemo_rl.experience.rollouts import run_async_multi_turn_rollout  # noqa: E402
from nemo_rl.models.generation import configure_generation_config  # noqa: E402
from nemo_rl.models.generation.vllm import VllmGeneration  # noqa: E402
from nemo_rl.utils.config import (  # noqa: E402
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def prompts_done(path: str, k: int) -> int:
    """Prompts fully scored in an earlier attempt (batches are written whole)."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        lines = sum(1 for _ in f)
    assert lines % k == 0, f"{path} has {lines} lines, not a multiple of {k}"
    return lines // k


def main() -> None:
    register_omegaconf_resolvers()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()
    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    score_cfg = (config.model_extra or {}).get("score") or {}
    k = int(score_cfg.get("num_generations", 8))
    batch_prompts = int(score_cfg.get("batch_prompts", 216))

    out_path = os.path.join(config.logger["log_dir"], "scores.jsonl")
    os.makedirs(config.logger["log_dir"], exist_ok=True)
    init_ray()

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    gen_cfg = configure_generation_config(config.policy["generation"], tokenizer)
    # As grpo.setup() does before it starts vLLM.
    gen_cfg["model_name"] = config.policy["model_name"]
    gen_cfg["_debug_payload_metrics"] = config.grpo.debug_payload_metrics
    gen_cfg["vllm_cfg"]["load_format"] = "auto"
    gen_cfg["vllm_cfg"]["precision"] = "bfloat16"  # keeps NeMo-RL's refit-only FP8 path off
    gen_cfg["colocated"]["enabled"] = False
    print("Generation config:")
    pprint.pprint(gen_cfg)

    dataset, _, task_to_env, _ = setup_response_data(tokenizer, config.data, config.env)
    done = prompts_done(out_path, k)
    print(f"{len(dataset)} prompts, {done} already scored in {out_path}, {k} answers each", flush=True)
    if done >= len(dataset):
        return
    loader = DataLoader(
        Subset(dataset, range(done, len(dataset))),
        batch_size=batch_prompts,
        shuffle=False,
        collate_fn=rl_collate_fn,
    )

    cl = config.cluster
    cluster = RayVirtualCluster(
        name="score_cluster",
        bundle_ct_per_node_list=[cl["gpus_per_node"]] * cl["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cl["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    t0 = time.perf_counter()
    generation = VllmGeneration(cluster=cluster, config=gen_cfg)
    print(f"vLLM up on {cl['num_nodes']} nodes in {time.perf_counter() - t0:.0f} s", flush=True)

    scored, correct, start = done, 0, time.perf_counter()
    for batch in loader:
        n_prompts = batch.size
        batch = batch.repeat_interleave(k)
        t = time.perf_counter()
        batch, metrics = run_async_multi_turn_rollout(
            generation,
            batch,
            tokenizer,
            task_to_env,
            max_seq_len=config.policy["max_total_sequence_length"],
            max_rollout_turns=config.grpo.max_rollout_turns,
            greedy=False,
        )
        rewards = batch["total_reward"].tolist()
        with open(out_path, "a") as f:
            for i, (log, r) in enumerate(zip(batch["message_log"], rewards)):
                asst = [m for m in log if m["role"] == "assistant"]
                spec = json.loads(batch["extra_env_info"][i]["ground_truth"])
                row = {
                    "idx": scored + i // k,
                    "task": spec["task"],
                    "difficulty": spec["difficulty"],
                    "reward": r,
                    "prompt_tokens": int(len(log[0]["token_ids"])),
                    "gen_tokens": int(sum(len(m["token_ids"]) for m in asst)),
                    "text": "".join(str(m["content"]) for m in asst),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        scored += n_prompts
        correct += sum(1 for r in rewards if r > 0.5)
        rate = (scored - done) / (time.perf_counter() - start)
        batch_acc = sum(1 for r in rewards if r > 0.5) / len(rewards)
        mean_tokens = metrics.get("mean_gen_tokens_per_sample", float("nan"))
        print(
            f"scored {scored}/{len(dataset)} prompts | batch {time.perf_counter() - t:.0f} s, "
            f"accuracy {batch_acc:.3f}, mean tokens {mean_tokens:.0f} | "
            f"ETA {(len(dataset) - scored) / rate / 60:.0f} min",
            flush=True,
        )

    print(f"done: {scored} prompts, accuracy {correct / ((scored - done) * k):.3f}", flush=True)
    generation.shutdown()
    shutdown_environments(task_to_env)


if __name__ == "__main__":
    main()
