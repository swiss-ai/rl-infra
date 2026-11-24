"""AdamW whose moments (m, v) live on scratch, not in host memory: the CPU optimizer for four-node Chonk GRPO.

Megatron's HybridDeviceOptimizer runs a torch optimizer over fp32 master copies in host memory. At 4 nodes Adam's m and v
alone would take about 217 GB per node (fournode/stage2/budget.py); ScratchAdamW keeps them in files on scratch and
streams them through a few aligned buffers during step(): readers prefetch the next chunks, the calling thread updates
the chunk, writers write it back, all overlapped. It is meant for HybridDeviceOptimizer(cpu_optimizer_cls=...) with
overlap_cpu_optimizer_d2h_h2d=False (one optimizer over all CPU parameters, so the whole step is one pipeline).

The update is torch's own, applied chunk by chunk, so results are bit-identical to torch.optim.AdamW: with fused=True
(what Megatron's CPU optimizer uses) the chunk goes through torch._fused_adamw_, otherwise through the single-tensor
operations of AdamW(foreach=False). Chunks start on 256-element boundaries, so vector blocks line up with the whole
tensor's.

State on scratch is double-buffered: a step reads generation g and writes generation 1-g into fresh files (new inodes),
then atomically replaces manifest.json. A crash mid-step leaves the previous generation and manifest intact. Because
every step writes new files, a hard link to a finished generation's files never changes afterwards: save_to() links
them into a checkpoint folder in no time and without copying.

The files are checked lazily, at the first step/save/restore/moments call. Megatron builds a throwaway
HybridDeviceOptimizer over the unsharded parameters before the distributed optimizer builds the real one, and that
throwaway must not trip over the state of the real one. The step counts in the optimizer state (which Megatron's
checkpoint loads) must equal the steps recorded with the files; step() refuses to run otherwise, so stale moments can
never be paired with another step's weights.

Megatron integration (patch 0007 in apertus2/patches/megatron): OptimizerConfig.cpu_optimizer_state_dir makes
get_megatron_optimizer use bind(<dir>, "rank<r>/hdo<i>") as the CPU optimizer class; HybridDeviceOptimizer.dummy_step()
calls skip_next_step(). NeMo-RL (patch 0034) calls save_scratch_state / restore_scratch_state / reset_scratch_state.
"""
from __future__ import annotations

import json
import mmap
import os
import queue
import shutil
import threading
import time
from pathlib import Path

import torch

ALIGN = 4096
CHECKPOINT_SUBDIR = "optimizer-scratch"


def _aligned(nbytes):
    return (nbytes + ALIGN - 1) // ALIGN * ALIGN


def _link_or_copy(src, dst):
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:          # another file system (EXDEV) or no hard links: copy
        shutil.copyfile(src, dst)


def _write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


class ScratchAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, *, state_dir, name=None,
                 chunk_numel=64 << 20, n_files=8, buffers=6, readers=3, writers=3, direct=True,
                 maximize=False, amsgrad=False, foreach=None, fused=None, **_ignored):
        if amsgrad or maximize:
            raise ValueError("ScratchAdamW supports neither amsgrad nor maximize")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=bool(fused)))
        self.state_dir = Path(state_dir)
        self.name = name if name is not None else self.state_dir.name
        self.chunk_numel = (int(chunk_numel) + 255) // 256 * 256
        self.n_files = int(n_files)
        self.n_buffers = int(buffers)
        self.n_readers = int(readers)
        self.n_writers = int(writers)
        self.direct = bool(direct) and hasattr(os, "O_DIRECT")
        self.stats = []
        self._layout()
        self.generation = None   # generation holding the current m, v; None: m = v = 0
        self._gen_steps = None   # step count of each parameter's moments in that generation (None: never stepped)
        self._loaded = False
        self._skip_next = False

    # --- layout: every parameter cut into chunks, chunks spread round-robin over n_files files -------------------

    def _params(self):
        return [p for g in self.param_groups for p in g["params"]]

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
        self.layout_sig = [list(p.shape) for p in self._params()] + [self.chunk_numel, self.n_files]

    def _path(self, generation, f):
        return self.state_dir / f"g{generation}" / f"mv-{f:03d}.bin"

    def _ensure_loaded(self):
        """Read manifest.json (once). Step counts fill in only where the optimizer state has none."""
        if self._loaded:
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.state_dir / "manifest.json"
        if manifest.exists():
            m = json.loads(manifest.read_text())
            if m["layout"] != self.layout_sig:
                raise ValueError(f"{self.state_dir}: state written for another parameter layout")
            self.generation = m["generation"]
            self._gen_steps = list(m["steps"])
        else:
            self.generation = None
            self._gen_steps = [None] * len(self._params())
        for p, s in zip(self._params(), self._gen_steps):
            if s is not None and self._step_of(p) is None:
                self.state[p]["step"] = torch.tensor(float(s))
        self._loaded = True

    def _manifest(self, generation, steps):
        return {"generation": generation, "layout": self.layout_sig, "steps": steps, "name": self.name, "time": time.time()}

    def _step_of(self, p):
        """The parameter's step count, or None. self.state is a defaultdict: a lookup must not create an empty entry,
        Megatron reads s["step"] from every entry."""
        st = self.state.get(p)
        return None if st is None or "step" not in st else float(st["step"])

    def _drop_step(self, p):
        st = self.state.get(p)
        if st is not None:
            st.pop("step", None)
            if not st:
                del self.state[p]

    def _state_steps(self):
        return [self._step_of(p) for p in self._params()]

    def _check_steps(self, params_only=None):
        """The optimizer state's step counts must be the ones the files were written at."""
        for i, p in enumerate(self._params()):
            if params_only is not None and id(p) not in params_only:
                continue
            want = self._gen_steps[i] or 0.0
            have = self._step_of(p) or 0.0
            if want != have:
                raise RuntimeError(
                    f"ScratchAdamW {self.name}: m and v on scratch are at step {want:g} for parameter {i}, but the "
                    f"optimizer state says step {have:g}. Restore them from the checkpoint that state came from "
                    f"(restore_scratch_state) or start fresh (reset_scratch_state).")

    # --- I/O ------------------------------------------------------------------------------------------------------

    def _open(self, path, write):
        flags = (os.O_RDWR | os.O_CREAT | os.O_EXCL) if write else os.O_RDONLY
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

    def skip_next_step(self):
        """Make the next step() only create the step counts, without I/O or updates.

        HybridDeviceOptimizer.dummy_step() runs a step with zero gradients so that the optimizer state exists before a
        checkpoint load fills it. Here that step would shrink m and v on scratch and apply weight decay, so it is
        skipped; the checkpoint load sets the step counts afterwards."""
        self._skip_next = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._ensure_loaded()
        if self._skip_next:
            self._skip_next = False
            for p, s in zip(self._params(), self._gen_steps):
                self.state[p].setdefault("step", torch.tensor(float(s or 0.0)))
            return loss
        if all(it[0].grad is None for it in self.items):
            return loss
        self._check_steps({id(p) for p in self._params() if p.grad is not None})
        # every chunk passes through: one without a gradient is carried into the new generation unchanged
        work = self.items
        t0 = time.perf_counter()
        old, new = self.generation, (0 if self.generation is None else 1 - self.generation)
        (self.state_dir / f"g{new}").mkdir(exist_ok=True)
        for f in range(self.n_files):     # fresh inodes: checkpoints hold hard links to older generations' files
            self._path(new, f).unlink(missing_ok=True)
        # step counts advance once per parameter (as torch does), before its first chunk is updated
        stepped = {}
        for p, *_ in work:
            if p.grad is not None and id(p) not in stepped:
                stepped[id(p)] = p
                st = self.state[p]
                st["step"] = st.get("step", torch.tensor(0.0)) + 1
        buf_bytes = _aligned(8 * self.max_chunk)
        free = queue.Queue()
        for _ in range(min(self.n_buffers, len(work))):
            free.put(mmap.mmap(-1, buf_bytes))
        read_fds, write_fds = {}, {}
        loaded = [None] * len(work)
        ready = [threading.Event() for _ in work]
        to_write = queue.Queue()
        errors = []
        next_item = iter(range(len(work)))
        lock = threading.Lock()

        def reader():
            # The buffer comes before the item: the main loop consumes items in order, so a reader holding an item
            # must never wait for a buffer that readers of later items took (deadlock).
            while True:
                buf = free.get()
                with lock:
                    k = next(next_item, None)
                if k is None:
                    free.put(buf)
                    return
                if errors:      # a claimed item is always signalled, or the main loop would wait for it forever
                    free.put(buf)
                    ready[k].set()
                    continue
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

        threads = []
        compute = 0.0
        try:
            if old is not None:
                read_fds.update({f: self._open(self._path(old, f), False) for f in range(self.n_files)})
            write_fds.update({f: self._open(self._path(new, f), True) for f in range(self.n_files)})
            threads = [threading.Thread(target=reader, daemon=True) for _ in range(self.n_readers)]
            threads += [threading.Thread(target=writer, daemon=True) for _ in range(self.n_writers)]
            for t in threads:
                t.start()
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
        except Exception as error:  # noqa: BLE001 - e.g. opening a file failed
            errors.append(error)
        finally:
            if errors:      # unblock readers waiting for a buffer the stopped loop will never return
                for _ in range(self.n_readers):
                    free.put(mmap.mmap(-1, ALIGN))
            for _ in range(self.n_writers):
                to_write.put(None)
            for t in threads:
                t.join()
            for fd in write_fds.values():
                try:
                    os.fsync(fd)
                except OSError as error:
                    errors.append(error)
            for fd in list(read_fds.values()) + list(write_fds.values()):
                os.close(fd)
        if errors:
            # the manifest still names the old generation; the step counts roll back with it
            for p in stepped.values():
                self.state[p]["step"] -= 1
            raise RuntimeError(f"ScratchAdamW step failed, state left at generation {old}") from errors[0]
        self.generation = new
        self._gen_steps = self._state_steps()
        _write_json_atomic(self.state_dir / "manifest.json", self._manifest(new, self._gen_steps))
        seconds = time.perf_counter() - t0
        moved = sum(16 * (e - s) for _, _, s, e, _, _ in work) if old is not None else sum(8 * (e - s) for _, _, s, e, _, _ in work)
        self.stats.append({"seconds": seconds, "compute_seconds": compute, "params": sum(e - s for q, _, s, e, _, _ in work if q.grad is not None),
                           "io_GBps": moved / seconds / 1e9})
        return loss

    def _update(self, group, p, s, e, m, v):
        """torch's AdamW on one chunk: the fused CPU kernel, or _single_tensor_adamw's operations in its order."""
        beta1, beta2 = group["betas"]
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        param = p.view(-1)[s:e]
        grad = p.grad.view(-1)[s:e]
        if group.get("fused"):
            torch._fused_adamw_([param], [grad], [m], [v], [], [self.state[p]["step"]], amsgrad=False, lr=lr,
                                beta1=beta1, beta2=beta2, weight_decay=wd, eps=eps, maximize=False,
                                grad_scale=None, found_inf=None)
            return
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

    # --- checkpoints ----------------------------------------------------------------------------------------------

    def save_to(self, dst):
        """Hard-link the current generation's files and a manifest into dst (a checkpoint's folder for this optimizer)."""
        self._ensure_loaded()
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        for f in range(self.n_files):
            if self.generation is not None:
                _link_or_copy(self._path(self.generation, f), dst / f"mv-{f:03d}.bin")
            else:
                (dst / f"mv-{f:03d}.bin").unlink(missing_ok=True)
        m = self._manifest(None if self.generation is None else "here", self._gen_steps)
        _write_json_atomic(dst / "manifest.json", m)

    def restore_from(self, src):
        """Make the moments saved by save_to(src) the current ones (linked in as a new generation).

        The optimizer state's step counts, when set (a checkpoint load sets them), must match the saved ones."""
        self._ensure_loaded()
        src = Path(src)
        m = json.loads((src / "manifest.json").read_text())
        if m["layout"] != self.layout_sig:
            raise ValueError(f"{src}: moments saved for another parameter layout (other parallelism or model?)")
        steps = list(m["steps"])
        for i, p in enumerate(self._params()):
            have = self._step_of(p)
            if have is not None and have != (steps[i] or 0.0):
                raise RuntimeError(f"{src}: moments saved at step {steps[i]} for parameter {i}, but the loaded "
                                   f"optimizer state is at step {have:g}")
        if m["generation"] is None:
            new = None
        else:
            new = 0 if self.generation is None else 1 - self.generation
            (self.state_dir / f"g{new}").mkdir(parents=True, exist_ok=True)
            for f in range(self.n_files):
                _link_or_copy(src / f"mv-{f:03d}.bin", self._path(new, f))
        _write_json_atomic(self.state_dir / "manifest.json", self._manifest(new, steps))
        self.generation = new
        self._gen_steps = steps
        for p, s in zip(self._params(), steps):
            if s is not None:
                self.state[p]["step"] = torch.tensor(float(s))
            else:
                self._drop_step(p)

    def reset(self):
        """Start from m = v = 0 and step 0 (a fresh run, or a resume without optimizer state)."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "manifest.json").unlink(missing_ok=True)
        self.generation = None
        self._gen_steps = [None] * len(self._params())
        for p in self._params():
            self._drop_step(p)
        self._loaded = True

    # --- reading the moments back (tests, statistics) -------------------------------------------------------------

    def moments(self, p):
        """(m, v) of one parameter as in-memory tensors (zeros before its first step)."""
        self._ensure_loaded()
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


# --- glue for Megatron and NeMo-RL ----------------------------------------------------------------------------------

def bind(state_root, name, **options):
    """A ScratchAdamW subclass whose instances keep their state in <state_root>/<name>.

    HybridDeviceOptimizer builds its CPU optimizer as cpu_optimizer_cls(param_groups) and tests isinstance() against
    that class, so this is a class, not a functools.partial."""
    state_dir = Path(state_root) / name

    class BoundScratchAdamW(ScratchAdamW):
        def __init__(self, params, **kwargs):
            super().__init__(params, **{**options, **kwargs, "state_dir": state_dir, "name": name})

    BoundScratchAdamW.__name__ = BoundScratchAdamW.__qualname__ = f"ScratchAdamW[{name}]"
    return BoundScratchAdamW


def scratch_optimizers(optimizer):
    """Every ScratchAdamW inside a Megatron optimizer (ChainedOptimizer -> DistributedOptimizer -> HDO -> CPU)."""
    found, seen, todo = [], set(), [optimizer]
    while todo:
        o = todo.pop(0)
        if o is None or id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, ScratchAdamW):
            found.append(o)
            continue
        todo.extend(getattr(o, "chained_optimizers", None) or [])
        todo.append(getattr(o, "optimizer", None))
        todo.extend(getattr(o, "cpu_optimizers", None) or [])
    return found


def save_scratch_state(optimizer, checkpoint_dir):
    """Link every ScratchAdamW's current moments into <checkpoint_dir>/optimizer-scratch/<name>/ (this rank's)."""
    for o in scratch_optimizers(optimizer):
        o.save_to(Path(checkpoint_dir) / CHECKPOINT_SUBDIR / o.name)


def restore_scratch_state(optimizer, checkpoint_dir):
    """After a checkpoint load: make the moments saved with that checkpoint the current ones (this rank's)."""
    for o in scratch_optimizers(optimizer):
        src = Path(checkpoint_dir) / CHECKPOINT_SUBDIR / o.name
        if not (src / "manifest.json").exists():
            raise FileNotFoundError(
                f"{src}/manifest.json is missing: the checkpoint was saved without ScratchAdamW moments (another "
                f"optimizer or another parallel layout). Resume weights-only (checkpoint.load_optim false) instead.")
        o.restore_from(src)


def reset_scratch_state(optimizer):
    for o in scratch_optimizers(optimizer):
        o.reset()


def after_checkpoint_load(optimizer, load_dir):
    """Call after the training setup has (or has not) loaded a checkpoint into the optimizer.

    A checkpoint load sets the step counts (HybridDeviceOptimizer.dummy_step, then the loaded state): the moments
    saved with that checkpoint in <load_dir>/optimizer-scratch/ become the current ones. Without a load (fresh run,
    weights-only resume, fine-tuning) the moments start from zero. Returns "restored", "reset" or None (no
    ScratchAdamW in this optimizer)."""
    opts = scratch_optimizers(optimizer)
    if not opts:
        return None
    if any(s is not None for o in opts for s in o._state_steps()):
        if load_dir is None:
            raise RuntimeError("the optimizer state was loaded, but no checkpoint folder was given for its moments")
        restore_scratch_state(optimizer, load_dir)
        return "restored"
    reset_scratch_state(optimizer)
    return "reset"
