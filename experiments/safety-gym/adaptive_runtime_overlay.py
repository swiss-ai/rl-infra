"""Exact-source renderers for adaptive-only NeMo-RL compatibility corrections."""
import ast
import hashlib

EXPECTED_GRPO_SHA256 = "63a0dfa403a321e56b55e34078297486474fd294f2198d6c4d63e0930b963b20"
EXPECTED_COLLECTOR_SHA256 = "e9e6680148934199a2393ee409f88e91ded8d952c55b98c44a0483b457990039"

_GUARD = '''
def _adaptive_assert_nonempty_optimizer_minibatches(train_data, effective_global_batch_size):
    """Fail before update if invalid samples can fill a whole optimizer batch."""
    if type(effective_global_batch_size) is not int or effective_global_batch_size <= 0:
        raise ValueError("adaptive profile requires a positive effective global batch size")
    token_mask = train_data["token_mask"][:, 1:]
    sample_mask = train_data["sample_mask"]
    if token_mask.shape[0] != sample_mask.shape[0]:
        raise ValueError("adaptive sample and token masks have different row counts")
    row_count = int(sample_mask.numel())
    if row_count % effective_global_batch_size:
        raise ValueError(
            f"adaptive batch rows ({row_count}) are not divisible by effective global batch "
            f"size ({effective_global_batch_size}); fix rollout/global-batch divisibility"
        )
    zero_token_rows = int(
        ((token_mask * sample_mask.unsqueeze(-1)).sum(dim=-1) == 0).sum().item()
    )
    if zero_token_rows >= effective_global_batch_size:
        raise ValueError(
            f"adaptive batch has {zero_token_rows} zero-token rows for global batch "
            f"{effective_global_batch_size}; abort before optimizer update"
        )
'''


def _sha(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def render_adaptive_grpo(source: str) -> str:
    """Mask invalid GRPO baseline rows and guard zero-token optimizer batches."""
    digest = _sha(source)
    if digest != EXPECTED_GRPO_SHA256:
        raise ValueError(f"Unknown pinned GRPO source fingerprint: {digest}")
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compute_advantage"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "adv_estimator"
    ]
    if len(calls) != 2:
        raise ValueError(f"Expected two sync/async advantage calls, found {len(calls)}")
    if any(any(keyword.arg == "valid_mask" for keyword in call.keywords) for call in calls):
        raise ValueError("Pinned GRPO source already has an unexpected valid_mask change")
    baseline_anchor = '                        logprobs_reference=train_data.get("reference_policy_logprobs"),\n'
    if source.count(baseline_anchor) != 2:
        raise ValueError("Unknown sync/async advantage call layout")
    source = source.replace(
        baseline_anchor,
        baseline_anchor + '                        valid_mask=train_data["sample_mask"],\n',
    )

    train_call = "                    train_results = policy.train(\n"
    if source.count(train_call) != 3:
        raise ValueError("Unknown synchronous policy.train call layout")
    source = source.replace(
        train_call,
        '                    _adaptive_assert_nonempty_optimizer_minibatches(\n'
        '                        train_data, master_config.policy["train_global_batch_size"]\n'
        '                    )\n' + train_call,
        1,
    )
    async_none = """                    if train_rows is None:
                        train_results = policy.train(
"""
    async_none_new = """                    if train_rows is None:
                        _adaptive_assert_nonempty_optimizer_minibatches(
                            train_data, master_config.policy["train_global_batch_size"]
                        )
                        train_results = policy.train(
"""
    if source.count(async_none) != 1:
        raise ValueError("Unknown async no-selection policy.train call layout")
    source = source.replace(async_none, async_none_new, 1)
    async_selected = """                        train_results = policy.train(
                            trained_data,
                            loss_fn,
                            gbs=len(train_rows) // num_optimizer_steps,
"""
    async_selected_new = """                        if len(train_rows) % num_optimizer_steps:
                            raise ValueError(
                                "adaptive selected rows must divide evenly across optimizer steps"
                            )
                        effective_global_batch_size = len(train_rows) // num_optimizer_steps
                        _adaptive_assert_nonempty_optimizer_minibatches(
                            trained_data, effective_global_batch_size
                        )
                        train_results = policy.train(
                            trained_data,
                            loss_fn,
                            gbs=effective_global_batch_size,
"""
    if source.count(async_selected) != 1:
        raise ValueError("Unknown async selected-row policy.train call layout")
    source = source.replace(async_selected, async_selected_new, 1)

    tree = ast.parse(source)
    first_non_import = next(
        (node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))),
        None,
    )
    if first_non_import is None:
        raise ValueError("GRPO module has no insertion boundary")
    lines = source.splitlines(keepends=True)
    insert_at = first_non_import.lineno - 1
    source = "".join(lines[:insert_at]) + _GUARD + "\n" + "".join(lines[insert_at:])
    compile(source, "grpo.py", "exec")
    return source


def render_adaptive_collector(source: str) -> str:
    """Attest monotonic prompt ordinals only where the native collector stamps them."""
    digest = _sha(source)
    if digest != EXPECTED_COLLECTOR_SHA256:
        raise ValueError(f"Unknown pinned trajectory collector fingerprint: {digest}")
    anchor = "            stamped_row[NEMO_GYM_TASK_INDEX_KEY] = first_task_index + offset\n"
    if source.count(anchor) != 1:
        raise ValueError("Unknown native _stamp_task_indices assignment")
    stamped = anchor + '''            stamped_row["native_prompt_ordinal"] = first_task_index + offset
            stamped_row["native_prompt_ordinal_attested"] = True
'''
    source = source.replace(anchor, stamped, 1)
    compile(source, "trajectory_collector.py", "exec")
    return source
