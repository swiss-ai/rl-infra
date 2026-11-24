"""Standalone judge benchmark for the llm_judge environment.

Run through run.sh like a GRPO run (SCRIPT=<this file>): the job's nodes all go to env.judge_pool, started
with the same judge_pool code a training run uses, and the calls are built exactly as the llm_judge
environment builds them (its prompts, parsing and request fields). Task-file options:
  bench:
    triples: <jsonl with question, reference, answer>   # answer = the graded text (claim or boxed content)
    runs: [{mode: verdict, concurrency: 256, calls: 3000, temperature: 1.0, top_p: 1.0}, ...]
    chat_template_kwargs: {enable_thinking: false}
    max_tokens: 512        # justify
    top_logprobs: 20       # verdict
Writes <log_dir>/bench_<i>_<mode>_c<concurrency>.jsonl (one line per call: i, score, p_yes or label, latency,
judge text for justify) and <log_dir>/bench_summary.json (calls per second, latency percentiles, failures).
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, "/opt/nemo-rl")

import aiohttp  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from nemo_rl.algorithms.grpo import MasterConfig  # noqa: E402
from nemo_rl.distributed.virtual_cluster import init_ray  # noqa: E402
from nemo_rl.environments.judge_pool import judge_pool_address, start_judge_pool  # noqa: E402
from nemo_rl.environments.llm_judge_environment import (  # noqa: E402
    JUSTIFY_PROMPT,
    VERDICT_SYSTEM,
    VERDICT_USER,
    justify_label,
    verdict_probability,
)
from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers  # noqa: E402


def request(model, mode, t, run, bench):
    fields = dict(question=t["question"], expected_answer=t["reference"], generated_answer=t["answer"])
    body = {"model": model, "temperature": run["temperature"], "top_p": run["top_p"],
            "return_tokens_as_token_ids": False}
    if mode == "verdict":
        body["messages"] = [{"role": "system", "content": VERDICT_SYSTEM},
                            {"role": "user", "content": VERDICT_USER.format(**fields)}]
        body.update(max_tokens=1, logprobs=True, top_logprobs=bench.get("top_logprobs", 20))
    else:
        body["messages"] = [{"role": "user", "content": JUSTIFY_PROMPT.format(**fields)}]
        body["max_tokens"] = bench.get("max_tokens", 512)
    if bench.get("chat_template_kwargs"):
        body["chat_template_kwargs"] = dict(bench["chat_template_kwargs"])
    return body


async def one(session, url, body, mode):
    t0 = time.perf_counter()
    try:
        async with session.post(url + "/chat/completions", json=body) as r:
            r.raise_for_status()
            c = await r.json()
    except Exception as e:  # noqa: BLE001
        return {"failed": True, "error": f"{type(e).__name__}: {e}"[:200], "latency": time.perf_counter() - t0}
    out = {"failed": False, "latency": time.perf_counter() - t0,
           "prompt_tokens": (c.get("usage") or {}).get("prompt_tokens"),
           "completion_tokens": (c.get("usage") or {}).get("completion_tokens")}
    ch = c["choices"][0]
    if mode == "verdict":
        content = (ch.get("logprobs") or {}).get("content") or []
        p = verdict_probability(content[0].get("top_logprobs") or []) if content else None
        out.update(p_yes=p, top=[(e["token"], round(e["logprob"], 3)) for e in (content[0]["top_logprobs"][:5] if content else [])])
        out["score"] = p
    else:
        text = ch["message"].get("content") or ""
        lab = justify_label(text)
        out.update(label=lab, text=text[-1500:])
        out["score"] = None if lab is None else float(lab)
    return out


async def run_one(urls, model, triples, run, bench, path):
    sem = asyncio.Semaphore(run["concurrency"])
    n = min(run["calls"], len(triples))
    results = [None] * n
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                     timeout=aiohttp.ClientTimeout(total=600)) as session:
        async def task(i):
            async with sem:
                url = urls[i % len(urls)]
                results[i] = await one(session, url, request(model, run["mode"], triples[i], run, bench), run["mode"])
        t0 = time.perf_counter()
        await asyncio.gather(*[task(i) for i in range(n)])
        wall = time.perf_counter() - t0
    with open(path, "w") as f:
        for i, r in enumerate(results):
            f.write(json.dumps({"i": i, **r}) + "\n")
    lat = sorted(r["latency"] for r in results if not r["failed"])
    q = lambda p: lat[min(len(lat) - 1, int(p * (len(lat) - 1)))] if lat else None  # noqa: E731
    ctoks = [r.get("completion_tokens") or 0 for r in results if not r["failed"]]
    ptoks = [r.get("prompt_tokens") or 0 for r in results if not r["failed"]]
    return {**run, "calls_done": n, "wall_s": round(wall, 1), "calls_per_s": round(n / wall, 2),
            "failed": sum(r["failed"] for r in results), "no_verdict": sum((not r["failed"]) and r["score"] is None for r in results),
            "latency_p50": q(0.5), "latency_p95": q(0.95), "latency_max": lat[-1] if lat else None,
            "mean_prompt_tokens": round(sum(ptoks) / max(1, len(ptoks)), 1),
            "mean_completion_tokens": round(sum(ctoks) / max(1, len(ctoks)), 1), "file": os.path.basename(path)}


def add_rule_truth(triples):
    """For rule-checkable triples (gold), math-verify the answer against the reference, as HFVerifyWorker does."""
    import logging
    logging.getLogger("math_verify").setLevel(logging.CRITICAL)
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    f = math_metric(gold_extraction_target=(LatexExtractionConfig(),),
                    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()))
    for t in triples:
        if not t.get("gold"):
            continue
        try:
            score, _ = f(["\\boxed{" + t["reference"] + "}"], ["\\boxed{" + t["answer"] + "}"])
        except Exception:  # noqa: BLE001
            score = 0.0
        t["rule"] = float(score)


def main():
    register_omegaconf_resolvers()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()
    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    bench = (config.model_extra or {})["bench"]
    log_dir = config.logger["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    init_ray()
    env_configs = dict(config.env)
    t0 = time.perf_counter()
    start_judge_pool(env_configs, dict(config.cluster))
    urls, model = judge_pool_address("judge", 60)
    print(f"judge pool up in {time.perf_counter() - t0:.0f} s: {urls}", flush=True)
    triples = [json.loads(l) for l in open(bench["triples"])]
    if bench.get("skip_empty", True):
        triples = [t for t in triples if t.get("answer")]
    add_rule_truth(triples)
    with open(os.path.join(log_dir, "triples_used.jsonl"), "w") as f:
        for t in triples:
            f.write(json.dumps(t) + "\n")
    summary = {"model": model, "replicas": len(urls), "pool_start_s": round(time.perf_counter() - t0), "runs": []}
    for i, run in enumerate(bench["runs"]):
        path = os.path.join(log_dir, f"bench_{i}_{run['mode']}_c{run['concurrency']}.jsonl")
        res = asyncio.run(run_one(urls, model, triples, run, bench, path))
        print(json.dumps(res), flush=True)
        summary["runs"].append(res)
        with open(os.path.join(log_dir, "bench_summary.json"), "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
