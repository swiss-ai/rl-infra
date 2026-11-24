#!/usr/bin/env python3
"""ScratchAdamW inside Megatron's HybridDeviceOptimizer on one GH200 GPU (stage 2; hdo_job.sbatch runs it).

What the laptop tests cannot cover: bf16 CUDA params with pinned fp32 host copies, CUDA streams, Megatron's param
groups, O_DIRECT and hard links on Lustre, aarch64 vector kernels. Uses the patched hybrid_optimizer.py (patches
0001-0007) given with --hdo, and scratch_adam.py next to this file's folder.

  identical    4 steps: HDO + ScratchAdamW (fused) vs HDO + torch AdamW (fused, Megatron's CPU optimizer); GPU params
               and fp32 masters must be bit-identical, with a learning-rate change and two param groups
  resume       a requeue as Megatron does it: new params from the step-2 state, dummy_step(), load_state_dict(), the
               step counts set, after_checkpoint_load() from the linked step-2 moments; 2 more steps must equal the
               uninterrupted run, and the checkpoint links must be unchanged by later steps
  fresh        after_checkpoint_load() without a load resets the moments
  speed        one ScratchAdamW step and one torch AdamW step over --speed-m million params (O_DIRECT on scratch)

Prints one JSON line per check and exits 1 when one fails.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import scratch_adam  # noqa: E402

DEFAULTS = dict(lr=5e-6, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-14, bias_correction=True, fused=True)
SHAPES = [(1792, 1920), (3 * 1792,), (1920, 1792), (7,), (256, 4099)]


def load_hdo(path):
    spec = importlib.util.spec_from_file_location("hybrid_optimizer_patched", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HybridDeviceOptimizer


def make_params(seed, shapes=SHAPES):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(*s, generator=g, device="cuda").to(torch.bfloat16)) for s in shapes]


def groups(params):
    # two groups as Megatron makes them (weight decay / no weight decay), with its extra keys
    return [dict(params=params[:3], wd_mult=1.0, lr_mult=1.0, is_expert_parallel=True, is_decoupled_lr=False),
            dict(params=params[3:], wd_mult=0.0, lr_mult=1.0, is_expert_parallel=False, is_decoupled_lr=False,
                 weight_decay=0.0)]


def build(HDO, params, cpu_cls):
    return HDO(groups(params), offload_fraction=1.0, cpu_optimizer_cls=cpu_cls, gpu_optimizer_cls=torch.optim.AdamW,
               overlap_cpu_optimizer_d2h_h2d=False, pin_cpu_grads=True, pin_cpu_params=True,
               param_update_in_fp32=True, **DEFAULTS)


def set_grads(params, step):
    g = torch.Generator(device="cuda").manual_seed(1000 + step)
    for p in params:
        p.grad = (torch.randn(p.shape, generator=g, device="cuda") * 1e-3).to(torch.bfloat16)


def run_step(opt, params, step):
    if step == 3:
        for group in opt.param_groups:
            group["lr"] = 2e-6
    set_grads(params, step)
    opt.step()
    opt.zero_grad()
    torch.cuda.synchronize()


def masters(opt):
    return [opt.param_to_inner_param[p].detach().clone() for g in opt.param_groups for p in g["params"]]


def same(a, b):
    return all(torch.equal(x.cpu(), y.cpu()) for x, y in zip(a, b))


def digests(folder):
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(folder).glob("*.bin"))}


def check_identical_and_resume(HDO, root, report):
    ref_p = make_params(1)
    our_p = make_params(1)
    ref = build(HDO, ref_p, torch.optim.AdamW)
    cls = scratch_adam.bind(root / "live", "rank00000/hdo0", chunk_numel=1 << 20, direct=True)
    ours = build(HDO, our_p, cls)
    inner = scratch_adam.scratch_optimizers(ours)
    assert len(inner) == 1 and isinstance(inner[0], cls) and inner[0].param_groups[0].get("fused") is True
    per_step = []
    for step in (1, 2):
        run_step(ref, ref_p, step)
        run_step(ours, our_p, step)
        per_step.append(same(ref_p, our_p) and same(masters(ref), masters(ours)))
    # checkpoint at step 2, as NeMo-RL's save does it: Megatron's state (step counts + masters) and the linked moments
    saved_state = ours.state_dict()
    saved_params = [p.detach().clone() for p in our_p]
    ckpt = root / "step_2" / "policy" / "weights"
    scratch_adam.save_scratch_state(ours, ckpt)
    linked = digests(ckpt / "optimizer-scratch" / "rank00000/hdo0")
    for step in (3, 4):
        run_step(ref, ref_p, step)
        run_step(ours, our_p, step)
        per_step.append(same(ref_p, our_p) and same(masters(ref), masters(ours)))
    report({"check": "identical", "ok": all(per_step), "per_step": per_step,
            "scratch_step_s": [round(s["seconds"], 3) for s in scratch_adam.scratch_optimizers(ours)[0].stats]})

    # requeue: a new process would build new params from the checkpoint, then the optimizer, then load
    res_p = [torch.nn.Parameter(p.clone()) for p in saved_params]
    resumed = build(HDO, res_p, cls)
    live = root / "live" / "rank00000/hdo0"
    before = ((live / "manifest.json").read_text(), digests(live / "g0"), digests(live / "g1"))
    resumed.dummy_step()                         # DistributedOptimizer.load_state_dict on an empty state
    no_io = before == ((live / "manifest.json").read_text(), digests(live / "g0"), digests(live / "g1"))
    for v in resumed.state.values():
        v["step"] = torch.tensor(2.0)
    resumed.load_state_dict(saved_state)         # the HDO post hook builds a new ScratchAdamW
    outcome = scratch_adam.after_checkpoint_load(resumed, ckpt)
    for step in (3, 4):
        run_step(resumed, res_p, step)
    ok = outcome == "restored" and same(ref_p, res_p) and same(masters(ref), masters(resumed))
    unchanged = digests(ckpt / "optimizer-scratch" / "rank00000/hdo0") == linked
    report({"check": "resume", "ok": ok and unchanged and no_io, "outcome": outcome, "links_unchanged": unchanged,
            "dummy_step_no_io": no_io})

    # fresh: no load, so the moments must start from zero even though the live folder has step-4 files
    fresh_p = make_params(2)
    fresh = build(HDO, fresh_p, cls)
    outcome = scratch_adam.after_checkpoint_load(fresh, None)
    ref2_p = make_params(2)
    ref2 = build(HDO, ref2_p, torch.optim.AdamW)
    run_step(fresh, fresh_p, 1)
    run_step(ref2, ref2_p, 1)
    report({"check": "fresh", "ok": outcome == "reset" and same(fresh_p, ref2_p), "outcome": outcome})


def check_speed(HDO, root, report, millions):
    n = int(millions * 1e6)
    shapes = [(n // 4,)] * 4
    out = {"check": "speed", "params_m": millions}
    for label, cls in (("torch_adamw_fused", torch.optim.AdamW),
                       ("scratch_adamw", scratch_adam.bind(root / "speed", "rank00000/hdo0", direct=True))):
        params = make_params(3, shapes)
        opt = build(HDO, params, cls)
        times = []
        for step in (1, 2):
            set_grads(params, step)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.step()
            torch.cuda.synchronize()
            times.append(round(time.perf_counter() - t0, 2))
        out[label] = {"step_s": times}
        if label == "scratch_adamw":
            st = scratch_adam.scratch_optimizers(opt)[0].stats[-1]
            out[label].update(io_GBps=round(st["io_GBps"], 2), compute_s=round(st["compute_seconds"], 2))
        del opt, params
        gc.collect()                             # the HDO's hooks keep it in reference cycles
        torch.cuda.empty_cache()
    out["ok"] = True
    report(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdo", required=True, help="patched hybrid_optimizer.py")
    ap.add_argument("--root", required=True, help="folder for the state files (deleted at the end)")
    ap.add_argument("--speed-m", type=float, default=2000, help="million params for the speed check (0: skip)")
    args = ap.parse_args(argv)
    HDO = load_hdo(args.hdo)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    results = []

    def report(r):
        results.append(r)
        print(json.dumps(r), flush=True)

    for name, fn in (("identical+resume+fresh", lambda: check_identical_and_resume(HDO, root, report)),
                     ("speed", lambda: args.speed_m and check_speed(HDO, root, report, args.speed_m))):
        try:
            fn()
        except Exception:  # noqa: BLE001 - one failing check must not hide the others
            report({"check": name, "ok": False, "error": traceback.format_exc()})
    shutil.rmtree(root, ignore_errors=True)
    failed = [r["check"] for r in results if not r.get("ok")]
    print(json.dumps({"summary": "pass" if not failed else "fail", "failed": failed}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
