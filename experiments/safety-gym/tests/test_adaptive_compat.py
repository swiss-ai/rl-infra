import ast
import os
from pathlib import Path
import sys

import pytest

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
from adaptive_runtime_overlay import render_adaptive_collector, render_adaptive_grpo

RUNTIME = Path(os.environ.get("SAFETY_ADAPTIVE_RUNTIME", "/scratch/hanint17/safetygym-adaptive/src/nemo-rl-apertus2"))


def test_native_source_adapters_pin_expected_calls_and_stamps():
    if not (RUNTIME / "nemo_rl/algorithms/grpo.py").is_file():
        pytest.skip("feature-pinned native source is not staged in this test environment")
    grpo_source = (RUNTIME / "nemo_rl/algorithms/grpo.py").read_text()
    collector_source = (RUNTIME / "nemo_rl/algorithms/async_utils/trajectory_collector.py").read_text()
    grpo = render_adaptive_grpo(grpo_source)
    collector = render_adaptive_collector(collector_source)
    tree = ast.parse(grpo)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "compute_advantage"
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "adv_estimator"]
    assert len(calls) == 2
    assert all(any(k.arg == "valid_mask" for k in call.keywords) for call in calls)
    assert grpo.count("_adaptive_assert_nonempty_optimizer_minibatches(") == 4
    assert '"native_prompt_ordinal_attested"] = True' in collector
    assert '"native_prompt_ordinal"] = first_task_index + offset' in collector


def test_unknown_native_sources_fail_closed():
    with pytest.raises(ValueError, match="fingerprint"):
        render_adaptive_grpo("# changed runtime source\n")
    with pytest.raises(ValueError, match="fingerprint"):
        render_adaptive_collector("# changed collector\n")


def _native_guard():
    if not (RUNTIME / "nemo_rl/algorithms/grpo.py").is_file():
        pytest.skip("feature-pinned native source is not staged")
    import torch
    source = render_adaptive_grpo((RUNTIME / "nemo_rl/algorithms/grpo.py").read_text())
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "_adaptive_assert_nonempty_optimizer_minibatches")
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, "adaptive_grpo_guard.py", "exec"), namespace)
    return namespace[fn.name], torch


def test_guard_allows_one_masked_row_when_batch_has_valid_tokens():
    guard, torch = _native_guard()
    data = {"token_mask": torch.ones((3, 5)), "sample_mask": torch.tensor([1.0, 1.0, 0.0])}
    guard(data, 3)


def test_guard_aborts_all_invalid_minibatch_and_bad_divisibility():
    guard, torch = _native_guard()
    all_invalid = {"token_mask": torch.ones((4, 5)), "sample_mask": torch.zeros(4)}
    with pytest.raises(ValueError, match="zero-token rows"):
        guard(all_invalid, 4)
    bad_divisor = {"token_mask": torch.ones((3, 5)), "sample_mask": torch.ones(3)}
    with pytest.raises(ValueError, match="not divisible"):
        guard(bad_divisor, 2)


def test_native_grpo_baseline_excludes_invalid_rows():
    if not (RUNTIME / "nemo_rl/algorithms/grpo.py").is_file():
        pytest.skip("feature-pinned native source is not staged")
    sys.path.insert(0, str(RUNTIME))
    import torch
    from nemo_rl.algorithms.advantage_estimator import AdvEstimatorConfig, GRPOAdvantageEstimator
    from nemo_rl.algorithms.loss import ClippedPGLossConfig

    estimator = GRPOAdvantageEstimator(
        AdvEstimatorConfig(use_leave_one_out_baseline=False, normalize_rewards=True),
        ClippedPGLossConfig(),
    )
    prompt_ids = torch.tensor([[9], [9], [9]])
    mask = torch.ones((3, 4))
    valid_mask = torch.tensor([1.0, 1.0, 0.0])
    result = estimator.compute_advantage(
        prompt_ids=prompt_ids,
        rewards=torch.tensor([1.0, 0.0, -1.0]),
        mask=mask,
        valid_mask=valid_mask,
    )
    changed_invalid_reward = estimator.compute_advantage(
        prompt_ids=prompt_ids,
        rewards=torch.tensor([1.0, 0.0, 1_000_000.0]),
        mask=mask,
        valid_mask=valid_mask,
    )
    assert result[0, 0] > 0 and result[1, 0] < 0
    assert torch.allclose(result[0], -result[1])
    # Invalid rows are not required to have zero *computed* advantages: sample_mask
    # excludes them from policy loss. Their reward must not influence valid siblings.
    assert torch.allclose(result[:2], changed_invalid_reward[:2])


def test_native_grpo_singleton_and_all_invalid_groups_are_zero():
    if not (RUNTIME / "nemo_rl/algorithms/grpo.py").is_file():
        pytest.skip("feature-pinned native source is not staged")
    sys.path.insert(0, str(RUNTIME))
    import torch
    from nemo_rl.algorithms.advantage_estimator import AdvEstimatorConfig, GRPOAdvantageEstimator
    from nemo_rl.algorithms.loss import ClippedPGLossConfig

    estimator = GRPOAdvantageEstimator(
        AdvEstimatorConfig(use_leave_one_out_baseline=False, normalize_rewards=True),
        ClippedPGLossConfig(),
    )
    for valid in (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0])):
        result = estimator.compute_advantage(
            prompt_ids=torch.tensor([[5], [5]]),
            rewards=torch.tensor([0.5, -1.0]),
            mask=torch.ones((2, 3)),
            valid_mask=valid,
        )
        assert torch.equal(result, torch.zeros_like(result))
