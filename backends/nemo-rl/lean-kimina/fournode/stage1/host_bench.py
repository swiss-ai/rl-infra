#!/usr/bin/env python3
"""Stage-1 host measurements for the four-node Chonk GRPO design (one GH200 node, no model).

Each subcommand measures one primitive the design rests on and writes one JSON file:

  facts        node layout: NUMA memory, CPUs, GPUs, torch version
  c2c          pinned host <-> GPU copies over NVLink-C2C (one process per GPU, bound to its own Grace)
  gradsum      a micro-batch's bf16 gradients flushed to the host and summed into fp32 there, overlapped
  adam         torch AdamW on fp32 tensors held in host memory, per Grace socket (foreach and fused)
  io           scratch write and read speed with N parallel files, O_DIRECT (no page cache)
  stream-adam  one optimizer step over master weights, m and v that live on scratch: read a chunk, run Adam,
               write it back, with readers, compute and writers overlapped (the design's step 4)

The per-GPU and per-socket measurements run as one process per GPU/socket (launched by host_job.sbatch with
numactl); processes of one measurement meet at a common start time (--start-at, epoch seconds) so that the
"concurrent" numbers are concurrent. Every process writes <out>/<name>.<rank>.json; `merge` combines them.
Sizes default to the job's; tests use tiny ones (host_bench_test.py).
"""
import argparse
import json
import mmap
import os
import platform
import socket
import sys
import threading
import time
from pathlib import Path

GIB = 1 << 30
MIB = 1 << 20


def log(*parts):
    print(f"[host_bench {time.strftime('%H:%M:%S')}]", *parts, flush=True)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    tmp.replace(path)


def wait_until(epoch):
    if epoch:
        delay = epoch - time.time()
        if delay > 0:
            time.sleep(delay)


def rank_info():
    return {"rank": int(os.environ.get("BENCH_RANK", "0")), "host": socket.gethostname(),
            "numa_bind": os.environ.get("BENCH_NUMA", ""), "pid": os.getpid()}


# ---------------------------------------------------------------------------------------------------------------
# facts


def numa_nodes():
    out = []
    base = Path("/sys/devices/system/node")
    for d in sorted(base.glob("node[0-9]*"), key=lambda p: int(p.name[4:])) if base.exists() else []:
        info = {"node": int(d.name[4:])}
        try:
            info["cpus"] = (d / "cpulist").read_text().strip()
            for line in (d / "meminfo").read_text().splitlines():
                parts = line.split()
                if parts[2] in ("MemTotal:", "MemFree:", "FilePages:"):
                    info[parts[2].rstrip(":")] = int(parts[3]) * 1024
        except (OSError, IndexError):
            pass
        out.append(info)
    return out


def cmd_facts(args):
    import torch

    facts = {"python": platform.python_version(), "torch": torch.__version__, "cpus": os.cpu_count(),
             "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
             "numa": numa_nodes(), "cuda": torch.cuda.is_available(), "gpus": []}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            facts["gpus"].append({"index": i, "name": p.name, "total": total, "free": free})
    facts.update(rank_info())
    write_json(Path(args.out) / "facts.json", facts)
    log(json.dumps({k: v for k, v in facts.items() if k != "numa"}))


# ---------------------------------------------------------------------------------------------------------------
# c2c: copies between pinned host memory and the GPU


def timed_copies(dst, src, reps, stream):
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(reps):
            dst.copy_(src, non_blocking=True)
    stream.synchronize()
    return time.perf_counter() - t0


def cmd_c2c(args):
    import torch

    dev = torch.device("cuda", 0)          # CUDA_VISIBLE_DEVICES selects this process's GPU
    n = int(args.gb * GIB)
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    host.fill_(1)                           # first touch on this process's NUMA node (numactl --membind)
    pageable = torch.ones(n, dtype=torch.uint8)
    gpu = torch.empty(n, dtype=torch.uint8, device=dev)
    host2 = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    host2.fill_(1)
    gpu2 = torch.empty(n, dtype=torch.uint8, device=dev)
    s1, s2 = torch.cuda.Stream(dev), torch.cuda.Stream(dev)
    timed_copies(gpu, host, 1, s1)          # warm-up
    timed_copies(host, gpu, 1, s1)
    res = {"bytes": n, "reps": args.reps, **rank_info()}

    def both():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(s1):
            for _ in range(args.reps):
                gpu.copy_(host, non_blocking=True)
        with torch.cuda.stream(s2):
            for _ in range(args.reps):
                host2.copy_(gpu2, non_blocking=True)
        s1.synchronize(); s2.synchronize()
        return time.perf_counter() - t0

    # Phase times: solo phases one GPU at a time (rank r at start + r * slot), then all GPUs together.
    slot = args.slot
    wait_until(args.start_at + res["rank"] * slot if args.start_at else 0)
    res["solo_h2d_GBps"] = n * args.reps / timed_copies(gpu, host, args.reps, s1) / 1e9
    res["solo_d2h_GBps"] = n * args.reps / timed_copies(host, gpu, args.reps, s1) / 1e9
    res["solo_both_GBps_each_way"] = n * args.reps / both() / 1e9
    res["solo_pageable_h2d_GBps"] = n * args.reps / timed_copies(gpu, pageable, args.reps, s1) / 1e9
    wait_until(args.start_at + args.ranks * slot if args.start_at else 0)
    res["all_h2d_GBps"] = n * args.reps / timed_copies(gpu, host, args.reps, s1) / 1e9
    res["all_d2h_GBps"] = n * args.reps / timed_copies(host, gpu, args.reps, s1) / 1e9
    res["all_both_GBps_each_way"] = n * args.reps / both() / 1e9
    write_json(Path(args.out) / f"c2c.{res['rank']}.json", res)
    log("c2c", json.dumps(res))


# ---------------------------------------------------------------------------------------------------------------
# gradsum: bf16 gradients from the GPU summed into an fp32 host buffer, chunk by chunk, D2H overlapped with the add


def cmd_gradsum(args):
    import torch

    torch.set_num_threads(args.threads)
    dev = torch.device("cuda", 0)
    params = int(args.params_m * 1e6)
    chunk = int(args.chunk_mb * MIB // 2)  # bf16 elements per chunk
    grad = torch.randn(params, dtype=torch.bfloat16, device=dev)
    acc = torch.zeros(params, dtype=torch.float32)
    acc.fill_(0)                             # first touch here
    staging = [torch.empty(chunk, dtype=torch.bfloat16, pin_memory=True) for _ in range(2)]
    events = [torch.cuda.Event() for _ in range(2)]
    stream = torch.cuda.Stream(dev)
    res = {"params": params, "chunk_bytes": chunk * 2, "threads": args.threads, "reps": args.reps, **rank_info()}

    def cpu_add_only():
        src = torch.randn(params, dtype=torch.bfloat16)
        t0 = time.perf_counter()
        for _ in range(args.reps):
            acc.add_(src)
        return time.perf_counter() - t0

    def flush_and_sum():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.reps):
            starts = list(range(0, params, chunk))
            # issue the first copy, then: issue copy k+1, add chunk k on the CPU while it runs
            def issue(k):
                s = starts[k]
                e = min(s + chunk, params)
                with torch.cuda.stream(stream):
                    staging[k % 2][: e - s].copy_(grad[s:e], non_blocking=True)
                    events[k % 2].record(stream)
            issue(0)
            for k, s in enumerate(starts):
                if k + 1 < len(starts):
                    issue(k + 1)
                events[k % 2].synchronize()
                e = min(s + chunk, params)
                acc[s:e].add_(staging[k % 2][: e - s])
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    flush_and_sum()  # warm-up (pinned buffers, threads)
    wait_until(args.start_at)
    t = cpu_add_only()
    res["cpu_add_Gparams_per_s"] = params * args.reps / t / 1e9
    wait_until(args.start_at + args.slot if args.start_at else 0)
    t = flush_and_sum()
    res["flush_sum_Gparams_per_s"] = params * args.reps / t / 1e9
    res["flush_sum_bf16_GBps"] = params * 2 * args.reps / t / 1e9
    write_json(Path(args.out) / f"gradsum.{res['rank']}.json", res)
    log("gradsum", json.dumps(res))


# ---------------------------------------------------------------------------------------------------------------
# adam: torch AdamW over fp32 tensors in host memory


def adam_hparams(args):
    return dict(lr=args.lr, betas=(0.9, 0.999), eps=args.eps, weight_decay=args.weight_decay)


def cmd_adam(args):
    import torch

    torch.set_num_threads(args.threads)
    params = int(args.params_m * 1e6)
    per = int(args.tensor_m * 1e6)
    sizes = [min(per, params - s) for s in range(0, params, per)]
    ps = [torch.nn.Parameter(torch.randn(n)) for n in sizes]
    for p in ps:
        p.grad = torch.randn_like(p) * 1e-3
    res = {"params": params, "tensors": len(sizes), "threads": args.threads, "reps": args.reps, **rank_info()}
    wait_until(args.start_at)
    for variant in args.variants.split(","):
        kw = {"foreach": True} if variant == "foreach" else {"fused": True} if variant == "fused" else {"foreach": False}
        try:
            opt = torch.optim.AdamW(ps, **adam_hparams(args), **kw)
            opt.step()                       # creates m, v (first touch here), warm-up
            t0 = time.perf_counter()
            for _ in range(args.reps):
                opt.step()
            t = time.perf_counter() - t0
            res[f"{variant}_Gparams_per_s"] = params * args.reps / t / 1e9
            del opt
        except Exception as error:  # noqa: BLE001 - one variant failing must not lose the others
            res[f"{variant}_error"] = repr(error)
    write_json(Path(args.out) / f"adam.{res['rank']}.json", res)
    log("adam", json.dumps(res))


# ---------------------------------------------------------------------------------------------------------------
# io: O_DIRECT reads and writes with one thread per file


def aligned_buffer(nbytes):
    """Page-aligned anonymous memory (O_DIRECT needs aligned buffers, offsets and sizes)."""
    return mmap.mmap(-1, nbytes)


def open_file(path, write, direct):
    flags = os.O_WRONLY | os.O_CREAT if write else os.O_RDONLY
    if direct and hasattr(os, "O_DIRECT"):
        flags |= os.O_DIRECT
    return os.open(path, flags, 0o600)


class Meter:
    """Bytes moved, sampled once a second, to show stalls and the steady rate."""

    def __init__(self):
        self.lock = threading.Lock()
        self.total = 0
        self.samples = []
        self.stop = threading.Event()
        self.t0 = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def add(self, n):
        with self.lock:
            self.total += n

    def _run(self):
        last = 0
        while not self.stop.wait(1.0):
            with self.lock:
                now = self.total
            self.samples.append(now - last)
            last = now

    def finish(self):
        self.stop.set()
        self.thread.join()
        return time.perf_counter() - self.t0


def move_file(path, nbytes, block, write, direct, meter, buf):
    fd = open_file(path, write, direct)
    view = memoryview(buf)
    try:
        off = 0
        while off < nbytes:
            n = min(block, nbytes - off)
            done = os.pwritev(fd, [view[:n]], off) if write else os.preadv(fd, [view[:n]], off)
            if done <= 0:
                raise OSError(f"short {'write' if write else 'read'} at {off} of {path}")
            off += done
            meter.add(done)
        if write:
            os.fsync(fd)
    finally:
        os.close(fd)


def run_io(paths, nbytes, block, write, direct):
    meter = Meter()
    errors = []
    bufs = [aligned_buffer(block) for _ in paths]
    if write:
        rnd = os.urandom(block)          # incompressible data
        for b in bufs:
            b[:] = rnd

    def worker(i):
        try:
            move_file(paths[i], nbytes, block, write, direct, meter, bufs[i])
        except Exception as error:  # noqa: BLE001
            errors.append(repr(error))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(paths))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seconds = meter.finish()
    rates = sorted(s / 1e9 for s in meter.samples)
    return {"GBps": meter.total / seconds / 1e9, "seconds": seconds, "bytes": meter.total, "errors": errors,
            "p10_1s_GBps": rates[len(rates) // 10] if rates else None,
            "p50_1s_GBps": rates[len(rates) // 2] if rates else None}


def cmd_io(args):
    base = Path(args.dir)
    base.mkdir(parents=True, exist_ok=True)
    block = int(args.block_mb * MIB)
    results = {"block": block, "direct": not args.no_direct, "dir": str(base), "runs": [], **rank_info()}
    for streams in [int(s) for s in args.streams.split(",")]:
        nbytes = int(args.total_gb * GIB / streams) // block * block
        paths = [str(base / f"io-{streams}-{i}.bin") for i in range(streams)]
        run = {"streams": streams, "file_bytes": nbytes}
        run["write"] = run_io(paths, nbytes, block, True, not args.no_direct)
        log(f"io {base.name} {streams} streams write {run['write']['GBps']:.2f} GB/s")
        run["read"] = run_io(paths, nbytes, block, False, not args.no_direct)
        log(f"io {base.name} {streams} streams read {run['read']['GBps']:.2f} GB/s")
        if args.mixed:
            # half the files read while the other half are rewritten: the streamed optimizer's pattern
            half = streams // 2 or 1
            mixed = {}
            out = {}

            def go(key, ps, write):
                out[key] = run_io(ps, nbytes, block, write, not args.no_direct)
            t0 = time.perf_counter()
            ts = [threading.Thread(target=go, args=("read", paths[:half], False)),
                  threading.Thread(target=go, args=("write", paths[half:] or paths[:1], True))]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            mixed = {k: v["GBps"] for k, v in out.items()}
            mixed["seconds"] = time.perf_counter() - t0
            run["mixed"] = mixed
            log(f"io {base.name} {streams} streams mixed read {mixed['read']:.2f} + write {mixed['write']:.2f} GB/s")
        for p in paths:
            os.unlink(p)
        results["runs"].append(run)
        write_json(Path(args.out) / f"io-{args.tag}.json", results)


# ---------------------------------------------------------------------------------------------------------------
# stream-adam: master, m and v on scratch, one chunk file each [master | m | v] (fp32), updated in a pipeline


def barrier(base, name, rank, ranks, timeout=1800):
    """All ranks of a measurement touch base/name.<rank> and wait until every rank's file is there."""
    (Path(base) / f"{name}.{rank}").touch()
    t0 = time.time()
    while len(list(Path(base).glob(f"{name}.*"))) < ranks:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"barrier {name}: only {len(list(Path(base).glob(f'{name}.*')))} of {ranks} ranks")
        time.sleep(0.2)


class Pool:
    def __init__(self, items):
        self.cond = threading.Condition()
        self.items = list(items)

    def get(self):
        with self.cond:
            while not self.items:
                self.cond.wait()
            return self.items.pop()

    def put(self, item):
        with self.cond:
            self.items.append(item)
            self.cond.notify()


def verify_update(before, after, grad, hp, step):
    """The in-place update on the scratch buffer against torch's reference AdamW (non-fused) on copies."""
    import torch

    rp = torch.nn.Parameter(before[0].clone())
    rp.grad = grad
    opt = torch.optim.AdamW([rp], **hp, foreach=False)
    opt.state[rp] = {"step": torch.tensor(float(step - 1)), "exp_avg": before[1].clone(),
                     "exp_avg_sq": before[2].clone()}
    opt.step()
    st = opt.state[rp]
    out = {"master_changed": bool((after[0] != before[0]).any()),
           "max_abs_diff_master": float((after[0] - rp.detach()).abs().max()),
           "max_abs_diff_m": float((after[1] - st["exp_avg"]).abs().max()),
           "max_abs_diff_v": float((after[2] - st["exp_avg_sq"]).abs().max()),
           "max_update": float((rp.detach() - before[0]).abs().max())}
    out["ok"] = out["master_changed"] and out["max_abs_diff_master"] <= 1e-3 * max(out["max_update"], 1e-30) + 1e-12
    return out


def cmd_stream_adam(args):
    import queue

    import torch

    torch.set_num_threads(args.threads)
    base = Path(args.dir)
    base.mkdir(parents=True, exist_ok=True)
    rank = rank_info()["rank"]
    chunk = int(args.chunk_m * 1e6) // 1024 * 1024
    n_chunks = max(1, int(args.params_m * 1e6) // chunk)
    chunk_bytes = chunk * 12
    direct = not args.no_direct
    paths = [str(base / f"state-r{rank}-{k:04d}.bin") for k in range(n_chunks)]
    res = {"params": chunk * n_chunks, "chunks": n_chunks, "chunk_bytes": chunk_bytes, "buffers": args.buffers,
           "readers": args.readers, "writers": args.writers, "threads": args.threads, "direct": direct,
           "steps": [], **rank_info()}
    bufs = [aligned_buffer(chunk_bytes) for _ in range(args.buffers)]
    grad = torch.randn(chunk) * 1e-3     # stands in for the summed fp32 gradient of the chunk
    bf16_out = torch.empty(chunk, dtype=torch.bfloat16)  # the new weights for the GPUs / vLLM
    hp = adam_hparams(args)
    block = int(args.block_mb * MIB)

    def rw(path, buf, write):
        fd = open_file(path, write, direct)
        view = memoryview(buf)
        try:
            off = 0
            while off < chunk_bytes:
                n = min(block, chunk_bytes - off)
                done = os.pwritev(fd, [view[off:off + n]], off) if write else os.preadv(fd, [view[off:off + n]], off)
                if done <= 0:
                    raise OSError(f"short io at {off} of {path}")
                off += done
            if write:
                os.fsync(fd)
        finally:
            os.close(fd)

    # initial state: master random-ish, m = v = 0 (written with the same writers: the first write speed)
    wait_until(args.start_at)
    init = bufs[0]
    t = torch.frombuffer(init, dtype=torch.float32)
    t[:chunk].normal_()
    t[chunk:].zero_()
    t0 = time.perf_counter()
    work = queue.Queue()
    for p in paths:
        work.put(p)

    def init_writer():
        while True:
            try:
                p = work.get_nowait()
            except queue.Empty:
                return
            rw(p, init, True)
    ts = [threading.Thread(target=init_writer) for _ in range(args.writers)]
    for th in ts:
        th.start()
    for th in ts:
        th.join()
    res["init_write_GBps"] = chunk_bytes * n_chunks / (time.perf_counter() - t0) / 1e9
    log(f"stream-adam r{rank}: wrote {n_chunks} chunks at {res['init_write_GBps']:.2f} GB/s")

    for step in range(1, args.steps + 1):
        barrier(base, f"barrier-step{step}", rank, args.ranks)
        free = Pool(bufs)
        loaded = queue.Queue()
        to_write = queue.Queue()
        timing = {"read": 0.0, "compute": 0.0, "write": 0.0}
        lock = threading.Lock()
        todo = queue.Queue()
        for k in range(n_chunks):
            todo.put(k)
        errors = []

        def reader():
            while True:
                try:
                    k = todo.get_nowait()
                except queue.Empty:
                    return
                buf = free.get()
                t1 = time.perf_counter()
                try:
                    rw(paths[k], buf, False)
                except Exception as error:  # noqa: BLE001
                    errors.append(repr(error))
                with lock:
                    timing["read"] += time.perf_counter() - t1
                loaded.put((k, buf))

        def writer():
            while True:
                item = to_write.get()
                if item is None:
                    return
                k, buf = item
                t1 = time.perf_counter()
                try:
                    rw(paths[k], buf, True)
                except Exception as error:  # noqa: BLE001
                    errors.append(repr(error))
                with lock:
                    timing["write"] += time.perf_counter() - t1
                free.put(buf)

        t0 = time.perf_counter()
        rs = [threading.Thread(target=reader) for _ in range(args.readers)]
        ws = [threading.Thread(target=writer) for _ in range(args.writers)]
        for th in rs + ws:
            th.start()
        for _ in range(n_chunks):
            k, buf = loaded.get()
            t1 = time.perf_counter()
            flat = torch.frombuffer(buf, dtype=torch.float32)
            master, m, v = flat[:chunk], flat[chunk:2 * chunk], flat[2 * chunk:]
            ref = None
            if step == 1 and "verify" not in res:
                ref = [t.clone() for t in (master, m, v)]   # reference update on copies, checked below
            p = torch.nn.Parameter(master)          # shares the buffer: the update lands in place
            p.grad = grad
            opt = torch.optim.AdamW([p], **hp, fused=args.fused)
            opt.state[p] = {"step": torch.tensor(float(step - 1)), "exp_avg": m, "exp_avg_sq": v}
            opt.step()
            bf16_out.copy_(master)
            del opt, p
            if ref is not None:
                res["verify"] = verify_update(ref, (master, m, v), grad, hp, step)
            timing["compute"] += time.perf_counter() - t1
            to_write.put((k, buf))
        for th in rs:
            th.join()
        for _ in ws:
            to_write.put(None)
        for th in ws:
            th.join()
        seconds = time.perf_counter() - t0
        moved = 2 * chunk_bytes * n_chunks
        step_res = {"step": step, "seconds": seconds, "Gparams_per_s": chunk * n_chunks / seconds / 1e9,
                    "io_GBps_read_plus_write": moved / seconds / 1e9, "compute_seconds": timing["compute"],
                    "read_thread_seconds": timing["read"], "write_thread_seconds": timing["write"], "errors": errors}
        res["steps"].append(step_res)
        log(f"stream-adam r{rank} step {step}: {seconds:.1f} s, {step_res['Gparams_per_s']:.3f} Gparams/s, "
            f"io {step_res['io_GBps_read_plus_write']:.2f} GB/s, compute {timing['compute']:.1f} s")
        write_json(Path(args.out) / f"stream-adam.{rank}.json", res)
    # the barrier files stay: a rank that deletes its own while another still polls would block that rank forever
    # (host_job.sbatch removes the whole data folder at the end)
    for p in [] if args.keep else paths:
        try:
            os.unlink(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------------------------------------------
# merge: one JSON per measurement, node sums


def cmd_merge(args):
    out = Path(args.out)
    merged = {}
    for name in ("c2c", "gradsum", "adam", "stream-adam"):
        ranks = [json.loads(p.read_text()) for p in sorted(out.glob(f"{name}.[0-9]*.json"))]
        if ranks:
            merged[name] = ranks
    for p in sorted(out.glob("io-*.json")):
        merged[p.stem] = json.loads(p.read_text())
    if (out / "facts.json").exists():
        merged["facts"] = json.loads((out / "facts.json").read_text())
    write_json(out / "host.json", merged)
    log(f"merged {sorted(merged)} into {out / 'host.json'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-at", type=float, default=0.0, help="epoch seconds; processes of one test meet here")
    ap.add_argument("--slot", type=float, default=20.0, help="seconds per phase when phases are staggered")
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--threads", type=int, default=72)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--eps", type=float, default=1e-14)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("facts")
    p = sub.add_parser("c2c")
    p.add_argument("--gb", type=float, default=4.0)
    p.add_argument("--reps", type=int, default=5)
    p = sub.add_parser("gradsum")
    p.add_argument("--params-m", type=float, default=2000)
    p.add_argument("--chunk-mb", type=float, default=256)
    p.add_argument("--reps", type=int, default=3)
    p = sub.add_parser("adam")
    p.add_argument("--params-m", type=float, default=2000)
    p.add_argument("--tensor-m", type=float, default=256)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--variants", default="foreach,fused")
    p = sub.add_parser("io")
    p.add_argument("--dir", required=True)
    p.add_argument("--tag", default="default")
    p.add_argument("--streams", default="8,16,32")
    p.add_argument("--total-gb", type=float, default=128)
    p.add_argument("--block-mb", type=float, default=16)
    p.add_argument("--mixed", action="store_true")
    p.add_argument("--no-direct", action="store_true")
    p = sub.add_parser("stream-adam")
    p.add_argument("--dir", required=True)
    p.add_argument("--params-m", type=float, default=4000)
    p.add_argument("--chunk-m", type=float, default=128)
    p.add_argument("--buffers", type=int, default=6)
    p.add_argument("--readers", type=int, default=3)
    p.add_argument("--writers", type=int, default=3)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--block-mb", type=float, default=16)
    p.add_argument("--fused", action="store_true")
    p.add_argument("--no-direct", action="store_true")
    p.add_argument("--keep", action="store_true", help="tests: leave the chunk files")
    sub.add_parser("merge")
    args = ap.parse_args(argv)
    {"facts": cmd_facts, "c2c": cmd_c2c, "gradsum": cmd_gradsum, "adam": cmd_adam, "io": cmd_io,
     "stream-adam": cmd_stream_adam, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
