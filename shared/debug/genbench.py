"""Generation throughput of a run's vLLM setup on several nodes (one or more replicas), without the trainer.

Started by shared/debug/genbench.sh through the stack's launch.sh, so it runs where a training run's driver runs,
with the same config: `--config <a run's config.yaml> [overrides]`, as examples/run_grpo.py. The vLLM workers start
on dummy weights, then load the export from disk through the refit path (NeMo-RL patch 0046), as a run's first refit
does, FP8 quantization included.

Every request generates exactly MAX_TOKENS tokens (ignore_eos, no stop token) after a prompt of random token ids, and
all requests of a case are submitted at once, as rollouts are. Cases (BENCH_CASES, comma separated) are
"<requests per inference node>x<prompt tokens>". Results: one JSON line per case in $OUT/genbench/results.jsonl.
Env: BENCH_NAME, BENCH_CASES (default 64x1024,128x1024,256x1024), MAX_TOKENS (6144).
"""
import argparse
import asyncio
import json
import os
import random
import time

import ray
import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.utils.config import load_config, parse_hydra_overrides

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args, overrides = parser.parse_known_args()
config = load_config(args.config)
if overrides:
    config = parse_hydra_overrides(config, overrides)
config = OmegaConf.to_container(config, resolve=True)

NAME = os.environ.get("BENCH_NAME", "genbench")
CASES = os.environ.get("BENCH_CASES", "64x1024,128x1024,256x1024").split(",")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "6144"))
OUT = os.path.join(os.path.dirname(config["logger"]["log_dir"].rstrip("/")), "..", "genbench")
OUT = os.path.realpath(OUT)
os.makedirs(OUT, exist_ok=True)
JOB = os.environ.get("SLURM_JOB_ID", time.strftime("t%m%d-%H%M"))

policy = config["policy"]
gen = policy["generation"]
gen["model_name"] = policy["model_name"]
gen["max_new_tokens"] = MAX_TOKENS
nodes = gen["colocated"]["resources"]["num_nodes"]
gpn = config["cluster"]["gpus_per_node"]
tokenizer = get_tokenizer(policy["tokenizer"])
gen = configure_generation_config(gen, tokenizer, is_eval=False)
gen["stop_token_ids"] = None
gen["stop_strings"] = None
gen["ignore_eos"] = True
vcfg = gen["vllm_cfg"]
setup = {"nodes": nodes, "tp": vcfg.get("tensor_parallel_size"), "ep": vcfg.get("expert_parallel_size"),
         "kv_cache_dtype": vcfg.get("kv_cache_dtype"), "max_model_len": vcfg.get("max_model_len"),
         "vllm_kwargs": gen.get("vllm_kwargs")}
print("SETUP", json.dumps(setup, default=str), flush=True)

init_ray()
cluster = RayVirtualCluster(name="genbench", bundle_ct_per_node_list=[gpn] * nodes, use_gpus=True,
                            num_gpus_per_node=gpn, max_colocated_worker_groups=1)
t = time.time()
vllm = VllmGeneration(cluster=cluster, config=gen)
setup["engine_up_s"] = round(time.time() - t)
t = time.time()
ok = ray.get([w.update_weights_from_hf_checkpoint_async.remote(model_path=gen["model_name"])
              for w in vllm._refit_leader_workers()])
setup["load_s"] = round(time.time() - t)
print("LOADED", json.dumps(setup, default=str), ok, flush=True)
assert all(ok), "loading the checkpoint through the refit path failed"


def prompt(salt, i, length):
    rng = random.Random(salt * 100003 + i)
    return [rng.randrange(1000, 30000) for _ in range(length)]


async def run(per_node, plen, salt):
    n = per_node * nodes
    prompts = [prompt(salt, i, plen) for i in range(n)]
    start = time.time()

    async def one(p):
        batch = BatchedDataDict({"input_ids": torch.tensor([p]), "input_lengths": torch.tensor([len(p)])})
        async for _, out in vllm.generate_async(batch):
            return time.time() - start, int(out["generation_lengths"][0])

    done = sorted(await asyncio.gather(*(one(p) for p in prompts)))
    wall = done[-1][0]
    tokens = sum(k for _, k in done)
    return {"name": NAME, "job": JOB, **setup, "requests_per_node": per_node, "requests": n, "prompt_tokens": plen,
            "max_tokens": MAX_TOKENS, "wall_s": round(wall, 1), "tokens": tokens,
            "tok_s": round(tokens / wall), "tok_s_per_node": round(tokens / wall / nodes),
            "stream_tok_s": round(MAX_TOKENS / wall, 1),
            "first_done_s": round(done[0][0], 1), "median_done_s": round(done[len(done) // 2][0], 1)}


async def main():
    r = await run(4, 256, 999)
    print("WARMUP", json.dumps(r, default=str), flush=True)
    for k, case in enumerate(CASES):
        per_node, plen = (int(x) for x in case.split("x"))
        r = await run(per_node, plen, k + 1)
        print("RESULT", json.dumps(r, default=str), flush=True)
        with open(os.path.join(OUT, "results.jsonl"), "a") as f:
            f.write(json.dumps({**r, "overrides": overrides}, default=str) + "\n")


asyncio.run(main())
vllm.shutdown()
