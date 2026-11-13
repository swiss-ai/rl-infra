"""ScienceWorld episodes with a served checkpoint, through NeMo-RL's own multi-turn rollout (no trainer).

Run through run.sh like a GRPO run (SCRIPT=<this file>, PREFLIGHT=0), so the config is a recipe + task + overrides.
The prompts are the task's training set, built by GRPO's setup_response_data (apertus2_sciworld_data_processor,
env scienceworld). vLLM loads the checkpoint itself (load_format auto, bf16). Each prompt is played
score.num_generations times through the rollout GRPO uses (run_sample_multi_turn_rollout), timed per episode.

Writes to <logger.log_dir>: episodes.jsonl (one line per episode: task, format, reward, the environment's counts,
turns, tokens per turn, think tokens per turn, wall time, transcript) and summary.json (per format and task).

Task-file options:
  score:
    num_generations: 4
    batch_prompts: 512
"""
import argparse
import asyncio
import json
import os
import pprint
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, "/opt/nemo-rl")

from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import nemo_rl.experience.rollouts as rollouts  # noqa: E402
from nemo_rl.algorithms.grpo import MasterConfig  # noqa: E402
from nemo_rl.algorithms.utils import get_tokenizer  # noqa: E402
from nemo_rl.data.collate_fn import rl_collate_fn  # noqa: E402
from nemo_rl.data.utils import setup_response_data  # noqa: E402
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray  # noqa: E402
from nemo_rl.environments.utils import shutdown_environments  # noqa: E402
from nemo_rl.models.generation import configure_generation_config  # noqa: E402
from nemo_rl.models.generation.vllm import VllmGeneration  # noqa: E402
from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers  # noqa: E402

_original = rollouts.run_sample_multi_turn_rollout


async def _timed(*args, **kwargs):
    start = time.time()
    state, metrics = await _original(*args, **kwargs)
    metrics["wall_s"] = time.time() - start
    metrics["start_s"] = start
    return state, metrics


rollouts.run_sample_multi_turn_rollout = _timed


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
    k = int(score_cfg.get("num_generations", 4))
    batch_prompts = int(score_cfg.get("batch_prompts", 512))
    out_dir = config.logger["log_dir"]
    os.makedirs(out_dir, exist_ok=True)
    init_ray()

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    gen_cfg = configure_generation_config(config.policy["generation"], tokenizer)
    gen_cfg["model_name"] = config.policy["model_name"]
    gen_cfg["_debug_payload_metrics"] = config.grpo.debug_payload_metrics
    gen_cfg["vllm_cfg"]["load_format"] = "auto"
    gen_cfg["vllm_cfg"]["precision"] = "bfloat16"
    gen_cfg["colocated"]["enabled"] = False
    print("Generation config:")
    pprint.pprint(gen_cfg)

    dataset, _, task_to_env, _ = setup_response_data(tokenizer, config.data, config.env)
    print(f"{len(dataset)} prompts, {k} episodes each", flush=True)
    loader = DataLoader(dataset, batch_size=batch_prompts, shuffle=False, collate_fn=rl_collate_fn)

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
    print(f"vLLM up in {time.perf_counter() - t0:.0f} s", flush=True)

    path = os.path.join(out_dir, "episodes.jsonl")
    rows = []
    for batch in loader:
        batch = batch.repeat_interleave(k)
        t = time.time()
        final, sample_metrics = asyncio.run(
            rollouts._run_multi_turn_rollout_async(
                policy_generation=generation,
                input_batch=batch,
                tokenizer=tokenizer,
                task_to_env=task_to_env,
                max_seq_len=config.policy["max_total_sequence_length"],
                max_rollout_turns=config.grpo.max_rollout_turns,
                greedy=False,
            )
        )
        batch_s = time.time() - t
        print(f"batch of {batch.size} episodes in {batch_s:.0f} s", flush=True)
        with open(path, "a") as f:
            for i, metrics in enumerate(sample_metrics):
                log = final["message_log"][i]
                info = final["extra_env_info"][i]
                spec = info["sciworld"]
                think_tokens = []
                transcript = []
                for message in log:
                    entry = {"role": message["role"], "content": message["content"],
                             "tokens": int(len(message["token_ids"]))}
                    parsed = message.get("apertus2_output")
                    if parsed is not None:
                        think = parsed.think or ""
                        think_tokens.append(len(tokenizer.encode(think, add_special_tokens=False)) if think else 0)
                    transcript.append(entry)
                row = {
                    "task": spec["task"], "variation": spec["variation"], "format": spec["action_format"],
                    "reward": float(final["total_reward"][i]),
                    "env": metrics.get("frame_metrics"),
                    "end": next((name.split("/", 1)[1] for name, value in (metrics.get("frame_metrics") or {}).items()
                                 if name.startswith("sw_end/") and value), None),
                    "turns": metrics["turn_count"], "truncated": bool(metrics["truncated"]),
                    "total_tokens": int(sum(len(m["token_ids"]) for m in log)),
                    "gen_tokens": metrics["assistant_tokens"], "env_tokens": metrics["env_tokens"],
                    "turn_gen_tokens": metrics["turn_gen_tokens"], "turn_input_tokens": metrics["turn_input_tokens"],
                    "think_tokens": think_tokens, "wall_s": metrics["wall_s"], "batch_s": batch_s,
                    "transcript": transcript,
                }
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summarize(rows, out_dir)
    for env in task_to_env.values():
        try:
            import ray
            print("worker stats:", ray.get(env.worker_stats.remote()), flush=True)
        except Exception as e:  # noqa: BLE001
            print("worker stats failed:", e)
    generation.shutdown()
    shutdown_environments(task_to_env)


def _mean(values):
    return statistics.mean(values) if values else float("nan")


def summarize(rows, out_dir) -> None:
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["format"], "ALL")].append(r)
        groups[(r["format"], r["task"])].append(r)
    for (fmt, task), rs in sorted(groups.items()):
        env = [r["env"] or {} for r in rs]
        out.setdefault(fmt, {})[task] = {
            "episodes": len(rs),
            "mean_reward": _mean([r["reward"] for r in rs]),
            "success": _mean([e.get("sw_success", 0.0) for e in env]),
            "zero_reward": _mean([float(r["reward"] <= 0) for r in rs]),
            "mean_turns": _mean([r["turns"] for r in rs]),
            "mean_actions": _mean([e.get("sw_actions", 0.0) for e in env]),
            "invalid_rate": _mean([e.get("sw_invalid_rate", 0.0) for e in env]),
            "invalid_per_episode": _mean([e.get("sw_invalid", 0.0) for e in env]),
            "mean_total_tokens": _mean([r["total_tokens"] for r in rs]),
            "mean_gen_tokens": _mean([r["gen_tokens"] for r in rs]),
            "mean_think_tokens_per_turn": _mean([t for r in rs for t in r["think_tokens"]]),
            "mean_gen_tokens_per_turn": _mean([t for r in rs for t in r["turn_gen_tokens"]]),
            "truncated": _mean([float(r["truncated"]) for r in rs]),
            "end": {reason: _mean([float(r["end"] == reason) for r in rs])
                    for reason in sorted({str(r["end"]) for r in rs})},
            "wall_s_mean": _mean([r["wall_s"] for r in rs]),
            "wall_s_max": max(r["wall_s"] for r in rs),
            "first_obs_mismatch": _mean([e.get("sw_first_obs_mismatch", 0.0) for e in env]),
            "replays": _mean([e.get("sw_replays", 0.0) for e in env]),
        }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    for fmt in out:
        print("SUMMARY", fmt, json.dumps(out[fmt]["ALL"], indent=1), flush=True)


if __name__ == "__main__":
    main()
