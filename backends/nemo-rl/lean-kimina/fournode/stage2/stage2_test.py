"""Local tests of stage 2's pieces: pytest stage2_test.py"""
import json
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
    with pytest.raises(ValueError, match="another parameter layout"):
        ScratchAdamW(make_params(2, SHAPES[:-1]), **HP, state_dir=tmp_path / "s", chunk_numel=700, direct=False)


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
    four = budget.budget(4, 2, "scratch-mv")
    assert four["gpu_fits"] and four["host_fits"]
    assert budget.budget(4, 3, "today") is None                     # 16 GPUs don't split into 3 stages
