"""Local tests of stage 2's pieces: pytest stage2_test.py"""
import json
import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import budget  # noqa: E402
import scratch_adam  # noqa: E402
from scratch_adam import ScratchAdamW  # noqa: E402

HP = dict(lr=5e-6, betas=(0.9, 0.999), eps=1e-14, weight_decay=0.01)


def make_params(seed, shapes):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(*s, generator=g)) for s in shapes]


def set_grads(params, seed, skip=()):
    g = torch.Generator().manual_seed(seed)
    for i, p in enumerate(params):
        p.grad = None if i in skip else torch.randn(p.shape, generator=g) * 1e-3


SHAPES = [(1000,), (37, 41), (5000,), (3,), (2048,)]


@pytest.mark.parametrize("chunk,n_files,buffers,readers,writers", [(512, 3, 2, 1, 1), (100000, 1, 1, 1, 1), (768, 8, 6, 3, 3)])
def test_bit_identical_to_torch_adamw(tmp_path, chunk, n_files, buffers, readers, writers):
    ref = make_params(0, SHAPES)
    ours = make_params(0, SHAPES)
    opt_ref = torch.optim.AdamW(ref, **HP, foreach=False)
    opt = ScratchAdamW(ours, **HP, state_dir=tmp_path / "s", chunk_numel=chunk, n_files=n_files, buffers=buffers,
                       readers=readers, writers=writers, direct=False)
    for step in range(1, 5):
        if step == 3:                       # a scheduler changing lr and weight decay between steps
            for o in (opt_ref, opt):
                o.param_groups[0]["lr"] = 2e-6
                o.param_groups[0]["weight_decay"] = 0.0
        skip = {1} if step == 2 else ()     # a parameter without a gradient in one step
        set_grads(ref, step, skip)
        set_grads(ours, step, skip)
        opt_ref.step()
        opt.step()
        for a, b in zip(ref, ours):
            assert torch.equal(a, b)
    for a, b in zip(ref, ours):
        m, v = opt.moments(b)
        assert torch.equal(m, opt_ref.state[a]["exp_avg"]) and torch.equal(v, opt_ref.state[a]["exp_avg_sq"])
        assert float(opt.state[b]["step"]) == float(opt_ref.state[a]["step"])
    assert opt.generation == 1 and json.loads((tmp_path / "s" / "manifest.json").read_text())["generation"] == 1


def test_restart_from_manifest_continues_identically(tmp_path):
    ref = make_params(1, SHAPES)
    a = make_params(1, SHAPES)
    opt_ref = torch.optim.AdamW(ref, **HP, foreach=False)
    opt = ScratchAdamW(a, **HP, state_dir=tmp_path / "s", chunk_numel=700, n_files=2, direct=False)
    for step in (1, 2):
        set_grads(ref, step)
        set_grads(a, step)
        opt_ref.step()
        opt.step()
    # a new process: parameters restored from a checkpoint, the optimizer from its state folder
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    opt2 = ScratchAdamW(b, **HP, state_dir=tmp_path / "s", chunk_numel=700, n_files=2, direct=False)
    opt2._ensure_loaded()
    assert opt2.generation == opt.generation
    set_grads(ref, 3)
    set_grads(b, 3)
    opt_ref.step()
    opt2.step()
    for x, y in zip(ref, b):
        assert torch.equal(x, y)


def test_layout_mismatch_is_refused(tmp_path):
    a = make_params(2, SHAPES)
    opt = ScratchAdamW(a, **HP, state_dir=tmp_path / "s", chunk_numel=700, direct=False)
    set_grads(a, 1)
    opt.step()
    # checked lazily: Megatron's throwaway optimizer over unsharded params shares the folder and is never used
    other = make_params(2, SHAPES[:-1])
    opt2 = ScratchAdamW(other, **HP, state_dir=tmp_path / "s", chunk_numel=700, direct=False)
    set_grads(other, 1)
    with pytest.raises(ValueError, match="another parameter layout"):
        opt2.step()


def test_failed_step_keeps_previous_generation(tmp_path, monkeypatch):
    a = make_params(3, SHAPES)
    ref = make_params(3, SHAPES)
    opt_ref = torch.optim.AdamW(ref, **HP, foreach=False)
    opt = ScratchAdamW(a, **HP, state_dir=tmp_path / "s", chunk_numel=300, n_files=2, direct=False)
    set_grads(a, 1)
    set_grads(ref, 1)
    opt.step()
    opt_ref.step()
    calls = {"n": 0}
    real_io = ScratchAdamW._io

    def flaky(self, fd, buf, nbytes, offset, write):
        calls["n"] += 1
        if write and calls["n"] > 5:
            raise OSError("disk full")
        return real_io(self, fd, buf, nbytes, offset, write)

    monkeypatch.setattr(ScratchAdamW, "_io", flaky)
    set_grads(a, 2)
    with pytest.raises(RuntimeError, match="generation 0"):
        opt.step()
    monkeypatch.setattr(ScratchAdamW, "_io", real_io)
    assert opt.generation == 0
    assert all(float(opt.state[p]["step"]) == 1.0 for p in a)
    # the moments of generation 0 are still step 1's
    for x, y in zip(ref, a):
        m, v = opt.moments(y)
        assert torch.equal(m, opt_ref.state[x]["exp_avg"]) and torch.equal(v, opt_ref.state[x]["exp_avg_sq"])


def test_chunks_round_to_vector_blocks(tmp_path):
    opt = ScratchAdamW(make_params(4, SHAPES), **HP, state_dir=tmp_path, chunk_numel=257, direct=False)
    assert opt.chunk_numel == 512


def test_rejects_non_fp32_or_gpu_params(tmp_path):
    with pytest.raises(ValueError, match="fp32 CPU"):
        ScratchAdamW([torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))], state_dir=tmp_path, direct=False)


def test_budget_calibration_and_four_nodes():
    today8 = budget.budget(8, 2, "today")
    assert round(today8["gpu_static"]) == 35                         # measured in job 4008441
    assert round(today8["host"]["Adam master+m+v (fp32)"]) == 162    # measured
    assert round(today8["host"]["fp32 grad copies (optimizer)"]) == 54
    assert not budget.budget(4, 2, "today")["host_fits"]
    assert not budget.budget(4, 2, "expert-offload")["host_fits"]
    # coarse-grained fp8 experts keep pinned fp8 copies too: with m, v on scratch the host still overflows at 4 nodes
    fp8 = budget.budget(4, 2, "scratch-mv")
    assert fp8["gpu_fits"] and not fp8["host_fits"] and round(fp8["host_total"]) == 491
    bf16_pp4 = budget.budget(4, 4, "bf16-scratch-mv")
    assert bf16_pp4["gpu_fits"] and bf16_pp4["host_fits"] and round(bf16_pp4["gpu_peak"][1]) == 89
    assert not budget.budget(4, 2, "bf16-scratch-mv")["gpu_fits"]
    c2c = budget.c2c_per_gpu(fp8)
    assert 4000 < c2c["H2D"] < 4400 and budget.c2c_per_gpu(bf16_pp4)["H2D"] < 20
    assert budget.budget(4, 3, "today") is None                     # 16 GPUs don't split into 3 stages


def file_digests(folder):
    import hashlib
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(folder).glob("*.bin"))}


def test_fused_bit_identical_to_torch_fused_adamw(tmp_path):
    ref = make_params(5, SHAPES)
    ours = make_params(5, SHAPES)
    opt_ref = torch.optim.AdamW(ref, **HP, fused=True)
    opt = ScratchAdamW(ours, **HP, fused=True, state_dir=tmp_path / "s", chunk_numel=512, n_files=3, direct=False)
    for step in range(1, 5):
        if step == 3:
            for o in (opt_ref, opt):
                o.param_groups[0]["lr"] = 2e-6
        skip = {2} if step == 2 else ()
        set_grads(ref, step, skip)
        set_grads(ours, step, skip)
        opt_ref.step()
        opt.step()
        for a, b in zip(ref, ours):
            assert torch.equal(a, b)
    for a, b in zip(ref, ours):
        m, v = opt.moments(b)
        assert torch.equal(m, opt_ref.state[a]["exp_avg"]) and torch.equal(v, opt_ref.state[a]["exp_avg_sq"])


def test_megatron_param_groups_extra_keys(tmp_path):
    """Megatron's groups carry bias_correction, fused and its own keys; fused=True selects the fused kernel."""
    ours = make_params(6, SHAPES[:2])
    ref = make_params(6, SHAPES[:2])
    group = dict(params=ours, lr=5e-6, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-14, bias_correction=True,
                 fused=True, wd_mult=1.0, lr_mult=1.0, is_expert_parallel=True)
    opt = ScratchAdamW([group], state_dir=tmp_path / "s", chunk_numel=256, direct=False)
    opt_ref = torch.optim.AdamW(ref, **HP, fused=True)
    for step in (1, 2):
        set_grads(ours, step)
        set_grads(ref, step)
        opt.step()
        opt_ref.step()
    assert all(torch.equal(a, b) for a, b in zip(ref, ours))


def test_checkpoint_links_survive_later_steps_and_restore(tmp_path):
    ref = make_params(7, SHAPES)
    a = make_params(7, SHAPES)
    opt_ref = torch.optim.AdamW(ref, **HP, foreach=False)
    opt = ScratchAdamW(a, **HP, state_dir=tmp_path / "live", chunk_numel=700, n_files=2, direct=False, name="rank00000/hdo0")
    for step in (1, 2):
        set_grads(ref, step)
        set_grads(a, step)
        opt_ref.step()
        opt.step()
    ckpt = tmp_path / "step_2" / "optimizer-scratch" / "rank00000/hdo0"
    opt.save_to(ckpt)
    saved = file_digests(ckpt)
    params_at_2 = [p.detach().clone() for p in a]
    assert all(os.stat(ckpt / n).st_ino == os.stat(opt._path(opt.generation, int(n[3:6]))).st_ino for n in saved)
    for step in (3, 4, 5):           # the live run goes on and reuses both generation folders
        set_grads(a, step)
        opt.step()
    assert file_digests(ckpt) == saved
    # a requeued job: params and step counts from the step-2 checkpoint, moments from its links
    b = [torch.nn.Parameter(p.clone()) for p in params_at_2]
    opt2 = ScratchAdamW(b, **HP, state_dir=tmp_path / "live", chunk_numel=700, n_files=2, direct=False, name="rank00000/hdo0")
    for p in b:
        opt2.state[p]["step"] = torch.tensor(2.0)          # what the checkpoint load sets
    with pytest.raises(RuntimeError, match="restore_scratch_state"):
        set_grads(b, 3)
        opt2.step()                  # the live files are at step 5: refused
    for p in b:
        assert float(opt2.state[p]["step"]) == 2.0
    opt2.restore_from(ckpt)
    for step in (3, 4):
        set_grads(ref, step)
        set_grads(b, step)
        opt_ref.step()
        opt2.step()
    assert all(torch.equal(x, y) for x, y in zip(ref, b))
    assert file_digests(ckpt) == saved


def test_restore_refuses_other_step(tmp_path):
    a = make_params(8, SHAPES)
    opt = ScratchAdamW(a, **HP, state_dir=tmp_path / "s", chunk_numel=700, direct=False)
    set_grads(a, 1)
    opt.step()
    opt.save_to(tmp_path / "ck")
    for p in a:
        opt.state[p]["step"] = torch.tensor(7.0)
    with pytest.raises(RuntimeError, match="saved at step 1"):
        opt.restore_from(tmp_path / "ck")


def test_skip_next_step_is_a_noop(tmp_path):
    a = make_params(9, SHAPES)
    before = [p.detach().clone() for p in a]
    opt = ScratchAdamW(a, **HP, state_dir=tmp_path / "s", chunk_numel=700, direct=False)
    opt.skip_next_step()
    for p in a:
        p.grad = torch.zeros_like(p)
    opt.step()
    assert all(torch.equal(x, y) for x, y in zip(before, a))
    assert all(float(opt.state[p]["step"]) == 0.0 for p in a)
    assert not list((tmp_path / "s").rglob("*.bin")) and not (tmp_path / "s" / "manifest.json").exists()
    opt.step()                      # the flag lasts one step
    assert all(float(opt.state[p]["step"]) == 1.0 for p in a)


def test_reset_starts_fresh(tmp_path):
    a = make_params(10, SHAPES)
    opt = ScratchAdamW(a, **HP, state_dir=tmp_path / "s", chunk_numel=700, direct=False)
    for step in (1, 2):
        set_grads(a, step)
        opt.step()
    ref = [torch.nn.Parameter(p.detach().clone()) for p in a]
    opt_ref = torch.optim.AdamW(ref, **HP, foreach=False)
    opt.reset()
    set_grads(a, 3)
    set_grads(ref, 3)
    opt.step()
    opt_ref.step()
    assert all(torch.equal(x, y) for x, y in zip(ref, a))


def test_bind_and_find_in_megatron_like_nesting(tmp_path):
    cls = scratch_adam.bind(tmp_path / "root", "rank00003/hdo1", chunk_numel=512, direct=False)
    params = make_params(11, SHAPES[:2])
    inner = cls([{"params": params, "lr": 1e-5}])
    assert isinstance(inner, cls) and isinstance(inner, ScratchAdamW)
    assert inner.state_dir == tmp_path / "root" / "rank00003/hdo1" and inner.chunk_numel == 512

    class Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    hdo = Obj(cpu_optimizers=[inner])
    chained = Obj(chained_optimizers=[Obj(optimizer=Obj(cpu_optimizers=[])), Obj(optimizer=hdo)])
    assert scratch_adam.scratch_optimizers(chained) == [inner]
    set_grads(params, 1)
    inner.step()
    scratch_adam.save_scratch_state(chained, tmp_path / "ck")
    assert (tmp_path / "ck/optimizer-scratch/rank00003/hdo1/manifest.json").exists()
    scratch_adam.restore_scratch_state(chained, tmp_path / "ck")
    with pytest.raises(FileNotFoundError, match="load_optim"):
        scratch_adam.restore_scratch_state(chained, tmp_path / "other")


def test_no_empty_state_entries(tmp_path):
    """Megatron's DistributedOptimizer.state_dict() reads s["step"] from every state entry."""
    a = make_params(12, SHAPES)
    opt = ScratchAdamW(a, **HP, state_dir=tmp_path / "s", chunk_numel=700, direct=False)
    set_grads(a, 1, skip={0})
    opt.step()
    opt._check_steps()
    opt.save_to(tmp_path / "ck")
    assert set(map(id, opt.state)) == {id(p) for p in a[1:]}
    assert all("step" in s for s in opt.state.values())
    opt.reset()
    assert len(opt.state) == 0


@pytest.mark.parametrize("buffers,readers,writers", [(2, 4, 1), (3, 3, 2), (1, 2, 1)])
def test_pipeline_never_deadlocks(tmp_path, buffers, readers, writers):
    """More readers than buffers and many small chunks: readers once claimed items before buffers and deadlocked."""
    import threading
    shapes = [(4096,)] * 6 + [(3000,)]
    ref = make_params(13, shapes)
    a = make_params(13, shapes)
    opt_ref = torch.optim.AdamW(ref, **HP, fused=True)
    opt = ScratchAdamW(a, **HP, fused=True, state_dir=tmp_path / "s", chunk_numel=256, n_files=4, buffers=buffers,
                       readers=readers, writers=writers, direct=False)
    done = []

    def run():
        for step in range(1, 26):
            set_grads(a, step)
            set_grads(ref, step)
            opt.step()
            opt_ref.step()
        done.append(True)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(60)
    assert done, "ScratchAdamW.step() hung"
    assert all(torch.equal(x, y) for x, y in zip(ref, a))


def test_after_checkpoint_load(tmp_path):
    a = make_params(14, SHAPES[:3])
    cls = scratch_adam.bind(tmp_path / "live", "rank00000/hdo0", chunk_numel=512, direct=False)
    opt = cls(a, **HP)
    for step in (1, 2):
        set_grads(a, step)
        opt.step()
    scratch_adam.save_scratch_state(opt, tmp_path / "ck")
    set_grads(a, 3)
    opt.step()
    # fresh setup without a load: moments start from zero, even though the live folder has step-3 files
    fresh = cls([torch.nn.Parameter(p.detach().clone()) for p in a], **HP)
    assert scratch_adam.after_checkpoint_load(fresh, tmp_path / "ck") == "reset"
    assert fresh.generation is None and not (tmp_path / "live/rank00000/hdo0/manifest.json").exists()
    # resume: the load set step 2 (dummy step + loaded state); the step-2 moments come back
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    resumed = cls(b, **HP)
    resumed.skip_next_step()
    resumed.step()
    for p in b:
        resumed.state[p]["step"] = torch.tensor(2.0)
    assert scratch_adam.after_checkpoint_load(resumed, tmp_path / "ck") == "restored"
    m, v = resumed.moments(b[0])
    assert float(v.abs().sum()) > 0
    assert scratch_adam.after_checkpoint_load(object(), None) is None


def test_megatron_patch_carries_this_scratch_adam():
    import patch_tool
    assert patch_tool.extract(patch_tool.PATCH.read_text()) == (HERE / "scratch_adam.py").read_text(), \
        "run: python3 patch_tool.py"
