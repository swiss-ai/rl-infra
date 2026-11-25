"""Router replay (R3) for Chonk's quantile-balancing routers: Megatron patch 0008 (patches/megatron).

Runs the patched TopKRouter._replay_qb_indices against Megatron's own RouterReplay, and against NeMo RL's fallback for
tokens without rollout routes (its real code). Needs the pinned sources, which are not in this repository:
  MLM_PIN_FILES  Megatron-LM-MoE files at the pin, in their repository paths (megatron/core/transformer/moe/router.py
                 and router_replay.py)
  NRL_FILES      NeMo RL files in their repository paths (nemo_rl/models/megatron/router_replay.py)
e.g. git -C ~/projects/megatron-lm-moe-apertus2 show <MEGATRON_COMMIT>:<path> > $MLM_PIN_FILES/<path>.
  python3 -m pytest -q router_replay_qb_test.py
"""
import ast
import importlib.util
import os
import subprocess
import sys
import types
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import pytest
import torch

HERE = Path(__file__).resolve().parent
PATCH = HERE / "patches" / "megatron" / "0008-router-replay-for-quantile-balancing.patch"
ROUTER = "megatron/core/transformer/moe/router.py"
REPLAY = "megatron/core/transformer/moe/router_replay.py"
NRL_REPLAY = "nemo_rl/models/megatron/router_replay.py"
MLM = os.environ.get("MLM_PIN_FILES")
NRL = os.environ.get("NRL_FILES")
needs_mlm = pytest.mark.skipif(not MLM, reason="MLM_PIN_FILES (pinned Megatron files) not set")
needs_nrl = pytest.mark.skipif(not (MLM and NRL), reason="MLM_PIN_FILES and NRL_FILES not set")
TOPK, EXPERTS, TOKENS = 8, 256, 64


def function_source(path, name):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(Path(path).read_text(), node)


@pytest.fixture
def patched_router(tmp_path):
    dst = tmp_path / ROUTER
    dst.parent.mkdir(parents=True)
    dst.write_text((Path(MLM) / ROUTER).read_text())
    subprocess.run(["git", "init", "-q", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "apply", "--include", ROUTER, str(PATCH)], cwd=tmp_path, check=True)
    return dst


@pytest.fixture
def replay_mod():
    """Pinned router_replay.py, importable under its Megatron name (NeMo RL's fallback imports it from there)."""
    name = "megatron.core.transformer.moe.router_replay"
    parts = name.split(".")
    saved = {k: sys.modules.get(k) for k in [".".join(parts[:i]) for i in range(1, len(parts) + 1)]}
    for i in range(1, len(parts)):
        sys.modules.setdefault(".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
    spec = importlib.util.spec_from_file_location(name, Path(MLM) / REPLAY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    yield mod
    mod.RouterReplay.clear_global_router_replay_instances()
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def make_router(patched_router, replay_mod):
    ns = {"torch": torch}
    exec(function_source(patched_router, "_replay_qb_indices"), ns)
    router = types.SimpleNamespace(router_replay=replay_mod.RouterReplay(), topk=TOPK)
    router.replay = types.MethodType(ns["_replay_qb_indices"], router)
    return router


def scores_and_qb(seed):
    g = torch.Generator().manual_seed(seed)
    biased = torch.randn(TOKENS, EXPERTS, generator=g)
    return biased, biased.topk(TOPK, dim=1).indices


def other_routes(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(EXPERTS, generator=g)[:TOPK] for _ in range(TOKENS)])


@needs_mlm
def test_hook_sits_after_the_beta_update_and_feeds_the_final_routing(patched_router):
    src = patched_router.read_text()
    qb = src[src.index("    def quantile_balancing("):src.index("    def _apply_aux_loss(")]
    hook = qb.index("indices = self._replay_qb_indices(biased_scores, indices)")
    assert qb.index("if should_update_beta:") < hook < qb.index("precomputed_indices=indices")


@needs_mlm
def test_no_action_keeps_qb_and_record_records_it(patched_router, replay_mod):
    router = make_router(patched_router, replay_mod)
    biased, qb = scores_and_qb(1)
    assert torch.equal(router.replay(biased, qb), qb)
    router.router_replay.set_router_replay_action(replay_mod.RouterReplayAction.RECORD)
    assert torch.equal(router.replay(biased, qb), qb)
    assert torch.equal(router.router_replay.get_recorded_indices(), qb)


@needs_mlm
def test_replay_forward_then_backward_use_the_rollout_routes(patched_router, replay_mod):
    router = make_router(patched_router, replay_mod)
    biased, qb = scores_and_qb(2)
    target = other_routes(3)
    router.router_replay.set_target_indices(target)
    router.router_replay.set_router_replay_action(replay_mod.RouterReplayAction.REPLAY_FORWARD)
    fwd = router.replay(biased, qb)
    assert torch.equal(fwd, target) and fwd.dtype == qb.dtype == torch.int64
    # activation recompute in the backward pass replays the same routes once
    router.router_replay.set_router_replay_action(replay_mod.RouterReplayAction.REPLAY_BACKWARD)
    assert torch.equal(router.replay(biased, qb), target)
    assert router.router_replay.replay_backward_list == []


@needs_nrl
def test_nemo_rl_fallback_takes_qb_for_tokens_without_routes(patched_router, replay_mod):
    path = Path(NRL) / NRL_REPLAY
    text = path.read_text()
    assert "_MISSING_ROUTE_SENTINEL = ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL" in text
    ns = {"inspect": __import__("inspect"), "wraps": wraps, "torch": torch, "Any": Any, "Optional": Optional,
          "_MISSING_ROUTE_SENTINEL": -1, "_MISSING_ROUTE_FALLBACK_PATCH_ATTR": "_nrl_missing_route_fallback_patch"}
    exec(function_source(path, "_install_missing_route_fallback_patch"), ns)
    original = replay_mod.RouterReplay.get_replay_topk
    try:
        ns["_install_missing_route_fallback_patch"]()
        assert replay_mod.RouterReplay.get_replay_topk is not original
        router = make_router(patched_router, replay_mod)
        biased, qb = scores_and_qb(4)
        target = other_routes(5)
        missing = torch.zeros(TOKENS, dtype=torch.bool)
        missing[::7] = True                      # e.g. the last generated token, which vLLM never routed
        target[missing] = -1
        router.router_replay.set_target_indices(target)
        router.router_replay.set_router_replay_action(replay_mod.RouterReplayAction.REPLAY_FORWARD)
        out = router.replay(biased, qb)
        assert torch.equal(out[missing], qb[missing]) and torch.equal(out[~missing], target[~missing])
        router.router_replay.set_router_replay_action(replay_mod.RouterReplayAction.REPLAY_BACKWARD)
        assert torch.equal(router.replay(biased, qb), out)   # the backward pass gets the same effective routes
    finally:
        replay_mod.RouterReplay.get_replay_topk = original
