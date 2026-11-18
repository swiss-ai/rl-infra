"""Validation of one checkpoint with vLLM only: no trainer, the run's own validation code.

The checkpoint, an HF export (convert_rl_ckpt.sh), is loaded into vLLM through the refit path, as a run's first refit
does (FP8 quantization included; NeMo-RL commit "vLLM: load an HF checkpoint from disk through the refit path").
Everything else comes from the saved config of the run that trained it: the message format, the environments and
their settings, the judge, the sampling (validation at temperature 1.0) and the limits. grpo.validate() then scores
the sets exactly as a periodic validation of the run would, multi-turn ScienceWorld and the open-QA judge included.

Runs as the driver command of ray.sub (submit_val.sh), inside the run's container, through driver.sh:
  run_val.py --model <HF export> --step <N> --name <W&B run name> [--sets a,b,...] [--limit ROWS_PER_SET]
Writes <log dir>/val_data_step<N>.jsonl (every answer with its reward) and <results dir>/<name>.json (metrics).
"""
import argparse
import json
import os
import time
from copy import deepcopy

import ray
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.grpo import MasterConfig, validate
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.environments.judge_pool import judge_pool_num_nodes, start_judge_pool
from nemo_rl.environments.utils import shutdown_environments
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.utils.config import register_omegaconf_resolvers
from nemo_rl.utils.logger import Logger

# The run whose settings the evaluation copies, the sets and the output folder; the defaults are run 5 at 32K.
RUN_CONFIG = os.environ.get("VAL_RUN_CONFIG", "/scratch/hanint06/nemo-rl/results/sftv02-stem-mix-run5-32k-49n/step_135/config.yaml")
BENCH = os.environ.get("BENCH_DIR", "/scratch/hanint06/data/benchmarks")
STEM_DATA = os.environ.get("STEM_DATA", "/scratch/hanint06/data/stem-rl")
LOGS = os.environ.get("VAL_LOGS", (os.environ.get("SCRATCH") or "/scratch/" + os.environ.get("USER", "")) + "/nemo-rl/logs/eval-run5-32k")

# Each set copies the run's training entry for the same kind of task (environment, processor, system frame) and
# replaces the data file. The file holds the repeats already, so every row is generated once.
SETS = {
    "aime25": ("run3/numina_clean.jsonl", "aime25_x32.jsonl"),
    "aime26": ("run3/numina_clean.jsonl", "aime26_x32.jsonl"),
    "math500": ("run3/numina_clean.jsonl", "math500_x4.jsonl"),
    "mmlu_pro": ("run3/mcqa_science_cs.jsonl", "mmlu_pro_x1.jsonl"),
    "reasoning_gym": ("reasoning_gym/train.jsonl", "reasoning_gym_val_x8.jsonl"),
    "enigmata": ("enigmata/train/enigmata_v1.jsonl", "enigmata_val_x2.jsonl"),
    "scienceworld": ("scienceworld/sciworld_act_train.jsonl", "sciworld_act_dev_x4.jsonl"),
    "openqa": ("openqa/openqa_science.jsonl", "openqa_val_x2.jsonl"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="HF export of the checkpoint")
ap.add_argument("--step", type=int, required=True, help="training step of the checkpoint (0 = start model)")
ap.add_argument("--name", required=True, help="W&B run name and results file name")
ap.add_argument("--sets", default=",".join(SETS))
ap.add_argument("--limit", type=int, default=0, help="debug: use only the first ROWS rows of every set")
ap.add_argument("--gen-nodes", type=int, required=True)
args = ap.parse_args()
sets = args.sets.split(",")

register_omegaconf_resolvers()
cfg = OmegaConf.to_container(OmegaConf.load(RUN_CONFIG), resolve=True)

# Data: the validation list becomes the evaluation sets; one tiny training entry keeps setup_response_data content.
train = {os.path.relpath(t.get("data_path", ""), STEM_DATA): t for t in cfg["data"]["train"]}
os.makedirs(f"{LOGS}/limited", exist_ok=True)
validation, rows = [], 0
for s in sets:
    src, path = SETS[s]
    path = f"{BENCH}/{path}"
    if args.limit:
        small = f"{LOGS}/limited/{os.path.basename(path)}"
        with open(path) as f, open(small, "w") as g:
            for i, line in enumerate(f):
                if i >= args.limit:
                    break
                g.write(line)
        path = small
    entry = deepcopy(train[src])
    entry["data_path"] = path
    validation.append(entry)
    with open(path) as f:
        rows += sum(1 for _ in f)
cfg["data"]["validation"] = validation
cfg["data"]["train"] = [deepcopy(validation[0])]
cfg["data"]["use_multiple_dataloader"] = False
cfg["data"]["shuffle"] = False

# Validation: every row once, in one batch, so the long answers of all sets overlap.
cfg["grpo"].update({"val_batch_size": rows, "max_val_samples": rows, "val_num_generations_per_prompt": 1,
                    "val_at_start": True, "val_period": 0})
judge_nodes = judge_pool_num_nodes(cfg["env"]) if "openqa" in sets else 0
if not judge_nodes:
    cfg["env"]["judge_pool"] = None
cfg["cluster"]["num_nodes"] = args.gen_nodes + judge_nodes

# Model and generation: the export through the refit path, on the generation nodes of this job.
cfg["policy"]["model_name"] = args.model
gen = cfg["policy"]["generation"]
gen["colocated"]["enabled"] = False
gen["colocated"]["resources"] = {"num_nodes": args.gen_nodes, "gpus_per_node": cfg["cluster"]["gpus_per_node"]}
gen["model_name"] = args.model

# Logging: W&B project chonk-rl, a folder of its own per evaluation.
log_dir = f"{LOGS}/{args.name}"
cfg["logger"]["log_dir"] = log_dir
cfg["logger"]["wandb"]["name"] = args.name
cfg["logger"]["monitor_gpus"] = False
cfg["checkpointing"]["enabled"] = False

master = MasterConfig(**cfg)
print(f"Evaluating {args.model} (step {args.step}) on {sets}: {rows} rows, {args.gen_nodes} vLLM nodes, "
      f"{judge_nodes} judge node(s)", flush=True)

init_ray()
tokenizer = get_tokenizer(master.policy["tokenizer"])
gen = configure_generation_config(master.policy["generation"], tokenizer)
_, val_dataset, task_to_env, val_task_to_env = setup_response_data(tokenizer, master.data, master.env)
val_loader = StatefulDataLoader(val_dataset, batch_size=rows, shuffle=False, collate_fn=rl_collate_fn,
                                num_workers=master.data["num_workers"])
if judge_nodes:
    start_judge_pool(master.env, master.cluster)

t0 = time.time()
# The name matters: ScienceWorld places its simulators on the placement group of the inference cluster.
cluster = RayVirtualCluster(name="grpo_inference_cluster", bundle_ct_per_node_list=[4] * args.gen_nodes,
                            use_gpus=True, num_gpus_per_node=4, max_colocated_worker_groups=1)
vllm = VllmGeneration(cluster=cluster, config=gen)
t1 = time.time()
ok = ray.get([w.update_weights_from_hf_checkpoint_async.remote(model_path=args.model)
              for w in vllm._refit_leader_workers()])
assert all(ok), f"loading {args.model} through the refit path failed: {ok}"
print(f"vLLM up in {t1 - t0:.0f} s, checkpoint loaded in {time.time() - t1:.0f} s", flush=True)

logger = Logger(master.logger)
t2 = time.time()
metrics, timings = validate(vllm, val_loader, tokenizer, val_task_to_env, step=args.step, master_config=master,
                            logger=logger)
elapsed = time.time() - t2
logger.log_metrics(metrics, args.step, prefix="validation")
logger.log_metrics(timings, args.step, prefix="timing/validation")

out = {"name": args.name, "model": args.model, "step": args.step, "sets": sets, "rows": rows,
       "gen_nodes": args.gen_nodes, "validation_s": round(elapsed, 1), "metrics": metrics, "timings": timings}
os.makedirs(f"{LOGS}/results", exist_ok=True)
with open(f"{LOGS}/results/{args.name}.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("RESULT " + json.dumps({k: v for k, v in metrics.items() if "accuracy" in k}, default=float), flush=True)
print(f"validation took {elapsed:.0f} s", flush=True)

shutdown_environments(task_to_env, val_task_to_env)
