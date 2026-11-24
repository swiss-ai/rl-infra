#!/usr/bin/env python3
"""Continuous-admission load on one vLLM server, and its sleep / wake timings (four-node design, stage 1).

  sweep      keeps C requests in flight per GPU (a finished request's slot is refilled at once, as the design's
             continuous admission does) for each level C in --levels, and measures generated tokens per second per
             GPU from vLLM's own counters over the last --measure seconds of each level. Prompts are the GRPO run's
             Lean rows, rendered exactly as the sampler and the training do (sampler.py's Apertus2Format), sent in
             groups of --group like GRPO's generations per prompt, at temperature 1.0 up to --max-model-len.
  sleepwake  after the sweep: waits for the server to drain, then times sleep level 1 / wake, sleep level 2 / wake
             / reload_weights (VLLM_SERVER_DEV_MODE endpoints), with GPU memory after each, and checks that the
             server still generates.

Outputs in --out: levels.json (one entry per level), metrics.jsonl (vLLM counters every --every s), requests.jsonl
(one line per finished request), sleepwake.json. The sweep stops admitting at --deadline (epoch) if given.
"""
import argparse
import http.client
import json
import random
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path


def log(*parts):
    print(f"[conc_client {time.strftime('%H:%M:%S')}]", *parts, flush=True)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    tmp.replace(path)


class Server:
    def __init__(self, url, timeout=3600):
        u = urllib.parse.urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.timeout = timeout

    def conn(self, timeout=None):
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)

    def request(self, method, path, body=None, timeout=60):
        c = self.conn(timeout)
        try:
            data = json.dumps(body).encode() if body is not None else None
            c.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
            r = c.getresponse()
            return r.status, r.read().decode(errors="replace")
        finally:
            c.close()


# vLLM's Prometheus counters and gauges, summed (counters, running/waiting) or averaged (cache use) over engines
WANTED = {
    "vllm:generation_tokens_total": "sum", "vllm:prompt_tokens_total": "sum",
    "vllm:num_requests_running": "sum", "vllm:num_requests_waiting": "sum",
    "vllm:gpu_cache_usage_perc": "mean", "vllm:kv_cache_usage_perc": "mean",
    "vllm:num_preemptions_total": "sum", "vllm:prefix_cache_hits_total": "sum",
    "vllm:prefix_cache_queries_total": "sum",
}


def parse_metrics(text):
    acc = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in WANTED:
            continue
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        acc.setdefault(name, []).append(value)
    out = {}
    for name, values in acc.items():
        out[name] = sum(values) if WANTED[name] == "sum" else sum(values) / len(values)
    if "vllm:kv_cache_usage_perc" in out:
        out["kv_usage"] = out["vllm:kv_cache_usage_perc"]
    elif "vllm:gpu_cache_usage_perc" in out:
        out["kv_usage"] = out["vllm:gpu_cache_usage_perc"]
    return out


def load_prompts(args):
    """Token IDs of the prompts and the request fields that make the sampler's requests GRPO-identical."""
    if args.fake_prompts:
        rng = random.Random(0)
        return [[rng.randrange(1000) for _ in range(rng.randrange(50, 300))] for _ in range(64)], {}
    sys.path.insert(0, args.sampler_dir)
    import sampler  # noqa: E402  (chonk-lean-sampling: Apertus2Format, STOP_TOKEN_IDS)

    fmt = sampler.Apertus2Format(args.tokenizer, args.system_prompt_file)
    rows = sampler.load_rows(args.data)
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(rows)), min(args.prompts, len(rows)))
    prompts = [fmt.encode(rows[i]["responses_create_params"]["input"]) for i in idx]
    extra = {"allowed_token_ids": fmt.allowed, "stop_token_ids": sampler.STOP_TOKEN_IDS, "skip_special_tokens": False}
    return prompts, extra


class Sweep:
    def __init__(self, args, prompts, extra):
        self.args = args
        self.server = Server(args.url, timeout=args.request_timeout)
        self.prompts = prompts
        self.extra = extra
        self.target = 0
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.next = 0
        self.level = None
        self.in_flight = 0
        self.out = Path(args.out)
        self.req_log = open(self.out / "requests.jsonl", "a")
        self.metrics_log = open(self.out / "metrics.jsonl", "a")
        self.latest = {}

    def take(self):
        with self.lock:
            k = self.next
            self.next += 1
        group = k // self.args.group
        return k, self.prompts[group % len(self.prompts)]

    def worker(self, i):
        while not self.stop.is_set():
            if i >= self.target:
                time.sleep(0.2)
                continue
            k, prompt = self.take()
            level = self.level
            body = {"model": self.args.model_name, "prompt": prompt,
                    "max_tokens": max(1, min(self.args.max_new_tokens, self.args.max_model_len - len(prompt))),
                    "temperature": self.args.temperature, "top_p": self.args.top_p, "return_token_ids": True,
                    **self.extra}
            t0 = time.time()
            with self.lock:
                self.in_flight += 1
            try:
                status, text = self.server.request("POST", "/v1/completions", body, timeout=self.args.request_timeout)
                rec = {"k": k, "level": level, "t0": t0, "t1": time.time(), "prompt": len(prompt), "status": status}
                if status == 200:
                    choice = json.loads(text)["choices"][0]
                    ids = choice.get("token_ids")
                    rec["gen"] = len(ids) if ids is not None else None
                    rec["finish"] = choice.get("finish_reason")
                else:
                    rec["error"] = text[:300]
            except Exception as error:  # noqa: BLE001 - a lost request is recorded, the slot refilled
                rec = {"k": k, "level": level, "t0": t0, "t1": time.time(), "error": repr(error)[:300]}
                time.sleep(1)
            finally:
                with self.lock:
                    self.in_flight -= 1
            with self.lock:
                self.req_log.write(json.dumps(rec) + "\n")
                self.req_log.flush()

    def metrics(self):
        while not self.stop.is_set():
            try:
                status, text = self.server.request("GET", "/metrics", timeout=30)
                m = parse_metrics(text) if status == 200 else {}
            except Exception as error:  # noqa: BLE001
                m = {"error": repr(error)[:200]}
            m.update({"t": time.time(), "level": self.level, "target": self.target, "in_flight": self.in_flight})
            self.latest = m
            self.metrics_log.write(json.dumps(m) + "\n")
            self.metrics_log.flush()
            self.stop.wait(self.args.every)

    def snapshot(self):
        status, text = self.server.request("GET", "/metrics", timeout=30)
        m = parse_metrics(text)
        m["t"] = time.time()
        return m

    def run(self):
        args = self.args
        levels = [int(x) for x in args.levels.split(",")]
        threading.Thread(target=self.metrics, daemon=True).start()
        for i in range(max(levels) * args.gpus):
            threading.Thread(target=self.worker, args=(i,), daemon=True).start()
        results_path = self.out / "levels.json"
        results = json.loads(results_path.read_text()) if results_path.exists() else []
        for level in levels:
            if args.deadline and time.time() + args.warm + args.measure > args.deadline:
                log(f"level {level}: not enough time before the deadline, stopping")
                break
            self.level = level
            self.target = level * args.gpus
            log(f"level {level} per GPU ({self.target} in flight): warming {args.warm} s")
            time.sleep(args.warm)
            a = self.snapshot()
            time.sleep(args.measure)
            b = self.snapshot()
            dt = b["t"] - a["t"]
            gen = b.get("vllm:generation_tokens_total", 0) - a.get("vllm:generation_tokens_total", 0)
            res = {"level_per_gpu": level, "in_flight_target": self.target, "seconds": dt,
                   "gen_tok_per_s": gen / dt, "gen_tok_per_s_per_gpu": gen / dt / args.gpus,
                   "prompt_tok_per_s": (b.get("vllm:prompt_tokens_total", 0) - a.get("vllm:prompt_tokens_total", 0)) / dt,
                   "preemptions": b.get("vllm:num_preemptions_total", 0) - a.get("vllm:num_preemptions_total", 0),
                   **self.window_means(a["t"], b["t"]), **self.window_requests(a["t"], b["t"])}
            results.append(res)
            write_json(results_path, results)
            log(f"level {level}: {res['gen_tok_per_s_per_gpu']:.0f} tok/s per GPU, running {res.get('mean_running')}, "
                f"KV use {res.get('mean_kv_usage')}, preemptions {res['preemptions']:.0f}")
        self.stop.set()
        return results

    def window_means(self, t0, t1):
        rows = []
        try:
            for line in open(self.out / "metrics.jsonl"):
                m = json.loads(line)
                if t0 <= m.get("t", 0) <= t1:
                    rows.append(m)
        except OSError:
            pass

        def mean(key):
            vals = [m[key] for m in rows if key in m]
            return sum(vals) / len(vals) if vals else None

        def peak(key):
            vals = [m[key] for m in rows if key in m]
            return max(vals) if vals else None

        return {"mean_running": mean("vllm:num_requests_running"), "mean_waiting": mean("vllm:num_requests_waiting"),
                "mean_kv_usage": mean("kv_usage"), "peak_kv_usage": peak("kv_usage"), "metric_samples": len(rows)}

    def window_requests(self, t0, t1):
        done, errors = [], 0
        with self.lock:
            self.req_log.flush()
        for line in open(self.out / "requests.jsonl"):
            r = json.loads(line)
            if t0 <= r.get("t1", 0) <= t1:
                if "gen" in r:
                    done.append(r)
                else:
                    errors += 1
        if not done:
            return {"finished": 0, "errors": errors}
        gens = sorted(r["gen"] for r in done)
        return {"finished": len(done), "mean_gen_tokens": sum(gens) / len(gens), "p50_gen_tokens": gens[len(gens) // 2],
                "cut_share": sum(r.get("finish") == "length" for r in done) / len(done),
                "mean_seconds": sum(r["t1"] - r["t0"] for r in done) / len(done),
                "errors": errors}


def cmd_sweep(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts, extra = load_prompts(args)
    log(f"{len(prompts)} prompts, mean {sum(map(len, prompts)) / len(prompts):.0f} tokens; groups of {args.group}")
    Sweep(args, prompts, extra).run()


def gpu_memory():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return [int(x) for x in out.split()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def cmd_sleepwake(args):
    server = Server(args.url)
    out = Path(args.out)
    res = {"steps": []}

    def record(name, method, path, body=None, timeout=1800):
        t0 = time.time()
        try:
            status, text = server.request(method, path, body, timeout=timeout)
        except Exception as error:  # noqa: BLE001 - every step is attempted and recorded
            status, text = None, repr(error)
        step = {"step": name, "status": status, "seconds": time.time() - t0, "reply": text[:300],
                "gpu_mem_mib": gpu_memory()}
        res["steps"].append(step)
        write_json(out / "sleepwake.json", res)
        log(f"{name}: status {status}, {step['seconds']:.1f} s, GPU MiB {step['gpu_mem_mib']}")
        return status

    def check():
        body = {"model": args.model_name, "prompt": [1, 2, 3, 4], "max_tokens": 16, "temperature": 0.0,
                "return_token_ids": True}
        return record("generate", "POST", "/v1/completions", body, timeout=300)

    t0 = time.time()
    while time.time() - t0 < args.drain:
        try:
            m = parse_metrics(server.request("GET", "/metrics", timeout=30)[1])
            if m.get("vllm:num_requests_running", 0) == 0 and m.get("vllm:num_requests_waiting", 0) == 0:
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    res["drain_seconds"] = time.time() - t0
    res["gpu_mem_mib_before"] = gpu_memory()
    check()
    record("sleep1", "POST", "/sleep?level=1")
    record("is_sleeping", "GET", "/is_sleeping", timeout=60)
    record("wake1", "POST", "/wake_up")
    check()
    record("sleep2", "POST", "/sleep?level=2")
    record("wake2", "POST", "/wake_up")
    record("reload_weights", "POST", "/collective_rpc", {"method": "reload_weights"})
    check()
    write_json(out / "sleepwake.json", res)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="chonk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sweep")
    p.add_argument("--levels", default="64,128,256", help="requests in flight per GPU")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--warm", type=float, default=480)
    p.add_argument("--measure", type=float, default=420)
    p.add_argument("--every", type=float, default=5)
    p.add_argument("--deadline", type=float, default=0)
    p.add_argument("--data")
    p.add_argument("--tokenizer")
    p.add_argument("--system-prompt-file")
    p.add_argument("--sampler-dir")
    p.add_argument("--prompts", type=int, default=512)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261124)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--request-timeout", type=float, default=3600)
    p.add_argument("--fake-prompts", action="store_true", help="tests: random token IDs, no tokenizer")
    p = sub.add_parser("sleepwake")
    p.add_argument("--drain", type=float, default=120)
    args = ap.parse_args(argv)
    {"sweep": cmd_sweep, "sleepwake": cmd_sleepwake}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
