"""AdamW whose moments (m, v) live on scratch, not in host memory: the CPU optimizer for four-node Chonk GRPO.

Megatron's HybridDeviceOptimizer runs a torch optimizer over fp32 master copies in host memory. At 4 nodes Adam's m and v
alone would take about 217 GB per node (fourth/stage2/budget.py); ScratchAdamW keeps them in files on scratch and
streams them through a few aligned buffers during step(): readers prefetch the next chunks, the calling thread updates
the chunk, writers write it back, all overlapped. It is meant for HybridDeviceOptimizer(cpu_optimizer_cls=...) with
overlap_cpu_optimizer_d2h_h2d=False (one optimizer over all CPU parameters, so the whole step is one pipeline).

The update is torch's single-tensor AdamW, operation for operation, so results are bit-identical to
torch.optim.AdamW(foreach=False) (tested). State on scratch is double-buffered: step N reads generation N%2 and writes
generation (N+1)%2, then atomically replaces manifest.json. A crash mid-step leaves the previous generation and manifest
intact, and the files of the last finished step are a consistent checkpoint of m and v.

Not here yet (stage 2 integration): checkpoint save/load through NeMo-RL (state_dict() materializes m and v only when
asked) and use of host-resident expert gradients without a copy.
"""
from __future__ import annotations

import json
import mmap
import os
import queue
import threading
import time
from pathlib import Path

import torch

ALIGN = 4096


def _aligned(nbytes):
    return (nbytes + ALIGN - 1) // ALIGN * ALIGN


class ScratchAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, *, state_dir,
                 chunk_numel=64 << 20, n_files=8, buffers=6, readers=3, writers=3, direct=True,
                 maximize=False, amsgrad=False, foreach=None, fused=None, **_ignored):
        if amsgrad or maximize:
            raise ValueError("ScratchAdamW supports neither amsgrad nor maximize")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # chunks start on 256-element boundaries: torch's CPU kernels vectorize in blocks from a tensor's start, and a
        # chunk starting mid-block would round its first elements differently (1 ulp) from the whole-tensor update
        self.chunk_numel = (int(chunk_numel) + 255) // 256 * 256
        self.n_files = int(n_files)
        self.n_buffers = int(buffers)
        self.n_readers = int(readers)
        self.n_writers = int(writers)
        self.direct = bool(direct) and hasattr(os, "O_DIRECT")
        self.stats = []
        self._layout()
        manifest = self.state_dir / "manifest.json"
        self.generation = None   # generation holding the current m, v; None: never stepped (m = v = 0)
        if manifest.exists():
            self._load_manifest(json.loads(manifest.read_text()))

    # --- layout: every parameter cut into chunks, chunks spread round-robin over n_files files -------------------

    def _layout(self):
        self.items = []          # (param, group index, start, end, file index, byte offset)
        offsets = [0] * self.n_files
        k = 0
        for gi, group in enumerate(self.param_groups):
            for p in group["params"]:
                if p.device.type != "cpu" or p.dtype != torch.float32:
                    raise ValueError(f"ScratchAdamW needs fp32 CPU parameters, got {p.dtype} on {p.device}")
                n = p.numel()
                for start in range(0, n, self.chunk_numel):
                    end = min(n, start + self.chunk_numel)
                    f = k % self.n_files
                    self.items.append((p, gi, start, end, f, offsets[f]))
                    offsets[f] += _aligned(8 * (end - start))
                    k += 1
        self.file_sizes = offsets
        self.max_chunk = max((e - s for _, _, s, e, _, _ in self.items), default=0)
        self.layout_sig = [list(p.shape) for g in self.param_groups for p in g["params"]] + [self.chunk_numel, self.n_files]

    def _path(self, generation, f):
        return self.state_dir / f"g{generation}" / f"mv-{f:03d}.bin"

    def _load_manifest(self, m):
        if m["layout"] != self.layout_sig:
            raise ValueError(f"{self.state_dir}: state written for another parameter layout")
        self.generation = m["generation"]
        for gi, group in enumerate(self.param_groups):
            for pi, p in enumerate(group["params"]):
                s = m["steps"][gi][pi]
                if s is not None:
                    self.state[p]["step"] = torch.tensor(float(s))

    def _write_manifest(self):
        steps = [[float(self.state[p]["step"]) if "step" in self.state[p] else None for p in g["params"]]
                 for g in self.param_groups]
        m = {"generation": self.generation, "layout": self.layout_sig, "steps": steps, "time": time.time()}
        tmp = self.state_dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(m))
        os.replace(tmp, self.state_dir / "manifest.json")

    # --- I/O ------------------------------------------------------------------------------------------------------

    def _open(self, path, write):
        flags = (os.O_RDWR | os.O_CREAT) if write else os.O_RDONLY
        if self.direct:
            flags |= os.O_DIRECT
        return os.open(path, flags, 0o600)

    def _io(self, fd, buf, nbytes, offset, write):
        view = memoryview(buf)
        done = 0
        while done < nbytes:
            n = (os.pwritev if write else os.preadv)(fd, [view[done:nbytes]], offset + done)
            if n <= 0:
                raise OSError(f"short {'write' if write else 'read'} at {offset + done}")
            done += n

    # --- the step -------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if all(it[0].grad is None for it in self.items):
            return loss
        # every chunk passes through: one without a gradient is carried into the new generation unchanged
        work = self.items
        t0 = time.perf_counter()
        old, new = self.generation, (0 if self.generation is None else 1 - self.generation)
        (self.state_dir / f"g{new}").mkdir(exist_ok=True)
        # step counts advance once per parameter (as torch does), before its first chunk is updated
        stepped = set()
        for p, *_ in work:
            if p.grad is not None and id(p) not in stepped:
                stepped.add(id(p))
                st = self.state[p]
                st["step"] = st.get("step", torch.tensor(0.0)) + 1
        buf_bytes = _aligned(8 * self.max_chunk)
        free = queue.Queue()
        for _ in range(min(self.n_buffers, len(work))):
            free.put(mmap.mmap(-1, buf_bytes))
        read_fds = {f: self._open(self._path(old, f), False) for f in range(self.n_files)} if old is not None else {}
        write_fds = {f: self._open(self._path(new, f), True) for f in range(self.n_files)}
        loaded = [None] * len(work)
        ready = [threading.Event() for _ in work]
        to_write = queue.Queue()
        errors = []
        next_item = iter(range(len(work)))
        lock = threading.Lock()

        def reader():
            while True:
                with lock:
                    k = next(next_item, None)
                if k is None or errors:
                    return
                buf = free.get()
                if errors:
                    return
                try:
                    _, _, s, e, f, off = work[k]
                    nb = _aligned(8 * (e - s))
                    if old is None:
                        buf[:nb] = bytes(nb)                      # first step: m = v = 0
                    else:
                        self._io(read_fds[f], buf, nb, off, False)
                except Exception as error:  # noqa: BLE001 - surfaced after the pipeline drains
                    errors.append(error)
                loaded[k] = buf
                ready[k].set()

        def writer():
            while True:
                item = to_write.get()
                if item is None:
                    return
                k, buf = item
                try:
                    _, _, s, e, f, off = work[k]
                    self._io(write_fds[f], buf, _aligned(8 * (e - s)), off, True)
                except Exception as error:  # noqa: BLE001
                    errors.append(error)
                free.put(buf)

        threads = [threading.Thread(target=reader, daemon=True) for _ in range(self.n_readers)]
        threads += [threading.Thread(target=writer, daemon=True) for _ in range(self.n_writers)]
        for t in threads:
            t.start()
        compute = 0.0
        try:
            for k, (p, gi, s, e, _, _) in enumerate(work):
                ready[k].wait()
                if errors:
                    break
                buf = loaded[k]
                loaded[k] = None
                c0 = time.perf_counter()
                n = e - s
                if p.grad is not None:
                    mv = torch.frombuffer(buf, dtype=torch.float32, count=2 * n)
                    self._update(self.param_groups[gi], p, s, e, mv[:n], mv[n:])
                compute += time.perf_counter() - c0
                to_write.put((k, buf))
        finally:
            if errors:      # unblock readers waiting for a buffer the stopped loop will never return
                for _ in range(self.n_readers):
                    free.put(mmap.mmap(-1, ALIGN))
            for _ in range(self.n_writers):
                to_write.put(None)
            for t in threads:
                t.join()
            for fd in list(read_fds.values()) + list(write_fds.values()):
                if fd in write_fds.values():
                    os.fsync(fd)
                os.close(fd)
        if errors:
            # the manifest still names the old generation; the step counts roll back with it
            for p in {id(it[0]): it[0] for it in work if it[0].grad is not None}.values():
                self.state[p]["step"] -= 1
            raise RuntimeError(f"ScratchAdamW step failed, state left at generation {old}") from errors[0]
        self.generation = new
        self._write_manifest()
        seconds = time.perf_counter() - t0
        moved = sum(16 * (e - s) for _, _, s, e, _, _ in work) if old is not None else sum(8 * (e - s) for _, _, s, e, _, _ in work)
        self.stats.append({"seconds": seconds, "compute_seconds": compute, "params": sum(e - s for q, _, s, e, _, _ in work if q.grad is not None),
                           "io_GBps": moved / seconds / 1e9})
        return loss

    def _update(self, group, p, s, e, m, v):
        """torch's _single_tensor_adamw on one chunk (same operations, same order)."""
        beta1, beta2 = group["betas"]
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        param = p.view(-1)[s:e]
        grad = p.grad.view(-1)[s:e]
        step = float(self.state[p]["step"])
        param.mul_(1 - lr * wd)
        m.lerp_(grad, 1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = lr / bias_correction1
        bias_correction2_sqrt = bias_correction2 ** 0.5
        denom = (v.sqrt() / bias_correction2_sqrt).add_(eps)
        param.addcdiv_(m, denom, value=-step_size)

    # --- reading the moments back (tests, conversion, checkpoint export) ---------------------------------------------

    def moments(self, p):
        """(m, v) of one parameter as in-memory tensors (zeros before its first step)."""
        m = torch.zeros(p.numel())
        v = torch.zeros(p.numel())
        if self.generation is None:
            return m.view_as(p), v.view_as(p)
        buf = mmap.mmap(-1, _aligned(8 * self.max_chunk))
        fds = {}
        try:
            for q, _, s, e, f, off in self.items:
                if q is not p:
                    continue
                if f not in fds:
                    fds[f] = self._open(self._path(self.generation, f), False)
                self._io(fds[f], buf, _aligned(8 * (e - s)), off, False)
                mv = torch.frombuffer(buf, dtype=torch.float32, count=2 * (e - s))
                m[s:e] = mv[: e - s]
                v[s:e] = mv[e - s:]
        finally:
            for fd in fds.values():
                os.close(fd)
        return m.view_as(p), v.view_as(p)
