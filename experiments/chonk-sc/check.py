"""Step 0 for runs A and B: resolve a composed config exactly as the SC driver does and validate it.

    python check.py <nemo_rl_dir> <composed.yaml> <expect: a|b> [overrides...]

Mirrors examples/run_grpo_single_controller.py up to setup: load_config, the launcher's
command-line overrides (parse_hydra_overrides), OmegaConf resolve, MasterConfig(**config) (pydantic
model validators), the entrypoint's async_grpo / data_plane checks, then NeMo-RL's own
validate_single_controller_config. The config-only checks of setup_single_controller that run
after it (validation off, data plane, checkpointing, data.use_multiple_dataloader, token capture
prerequisites) need the tokenizer and Ray there, so they are replicated below
(nemo_rl/algorithms/single_controller_utils/setup.py, setup_single_controller).

Paths under /opt/nemo-rl (the clone's mount point in the container) are read from <nemo_rl_dir>.
Packages that are not installed (torch, ray, vllm, transformers, megatron, ... on a login node) are
replaced by inert stubs so the config modules import; they are listed in the output. In the
container nothing is stubbed. CHONK_SC_STUB_ALL=1 stubs everything but the config dependencies
(to emulate a bare login-node venv).
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import types
import warnings

NEMO_RL_DIR, CFG, EXPECT, *OVERRIDES = sys.argv[1:]
EXPECT = EXPECT.lower()

# ── stubs for heavy runtime packages ────────────────────────────────────────────────────────────

# what the config code itself needs; never stubbed
_REAL = {
    "omegaconf", "hydra", "antlr4", "yaml", "pydantic", "pydantic_core", "typing_extensions",
    "annotated_types", "typing_inspection", "nemo_rl", "_distutils_hack", "packaging",
}
STUBBED: set[str] = set()


class _StubMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _stub_class(f"{cls.__name__}.{name}")

    def __getitem__(cls, item):
        return cls

    def __call__(cls, *args, **kwargs):
        # @decorator and @decorator(...) both hand the decorated object back unchanged
        if len(args) == 1 and not kwargs and (callable(args[0]) or isinstance(args[0], type)):
            return args[0]
        return cls

    def __or__(cls, other):
        return cls

    __ror__ = __or__

    def __iter__(cls):
        return iter(())

    def __bool__(cls):
        return False


def _stub_class(name: str) -> type:
    return _StubMeta(name, (), {"__module__": "stub"})


def _lock_versions() -> dict[str, str]:
    """Package versions pinned in the clone's uv.lock, for the stubs' __version__ (version gates)."""
    import re

    try:
        text = open(os.path.join(NEMO_RL_DIR, "uv.lock")).read()
    except OSError:
        return {}
    return {m[1]: m[2].split("+")[0] for m in re.finditer(r'^name = "([^"]+)"\nversion = "([^"]+)"', text, re.M)}


_LOCK = _lock_versions()


class _StubModule(types.ModuleType):
    pass

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        value = _stub_class(f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = _StubModule(spec.name)
        top = spec.name.split(".")[0]
        mod.__version__ = _LOCK.get(top) or _LOCK.get(top.replace("_", "-")) or "999.0.0"
        mod.__path__ = []  # a package, so `import x.y.z` works
        return mod

    def exec_module(self, module):
        STUBBED.add(module.__name__.split(".")[0])


def _importer() -> str:
    """Module name of the code running the current import statement."""
    f = sys._getframe(2)
    while f is not None:
        name = f.f_globals.get("__name__", "")
        if not (name.startswith(("importlib", "_frozen_importlib")) or name == __name__):
            return name
        f = f.f_back
    return ""


class _StubFinder(importlib.abc.MetaPathFinder):
    """Stubs a package that NeMo-RL (or another stub) imports and that is not installed.

    Only imports made by nemo_rl code or by stubs are considered, so the optional-dependency probes
    (try: import x / except ImportError) of real packages keep failing as they should. Last on
    sys.meta_path, so installed packages win; with CHONK_SC_STUB_ALL=1 first, so every non-config
    package nemo_rl imports is a stub even when installed.
    """

    def find_spec(self, fullname, path, target=None):
        top = fullname.split(".")[0]
        if top in _REAL or top in sys.stdlib_module_names:
            return None
        if top not in STUBBED and not _importer().startswith(("nemo_rl", "stub")):
            return None
        return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)


_stub_all = os.environ.get("CHONK_SC_STUB_ALL") == "1"
if _stub_all:
    sys.meta_path.insert(0, _StubFinder())
else:
    sys.meta_path.append(_StubFinder())
sys.path.insert(0, NEMO_RL_DIR)
warnings.simplefilter("always")
warnings.simplefilter("ignore", DeprecationWarning)
warnings.showwarning = lambda m, c, *a, **k: print(f"  WARN  {c.__name__}: {m}")

# ── load + resolve as the driver does ───────────────────────────────────────────────────────────
from omegaconf import OmegaConf  # noqa: E402

import nemo_rl.utils.config as nrl_config  # noqa: E402

_orig_resolve_path = nrl_config.resolve_path


def _resolve_path(base_path, path: str):
    if path.startswith("/opt/nemo-rl/") and not os.path.exists(path):
        path = os.path.join(NEMO_RL_DIR, path[len("/opt/nemo-rl/"):])
    return _orig_resolve_path(base_path, path)


nrl_config.resolve_path = _resolve_path
nrl_config.register_omegaconf_resolvers()

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


try:
    cfg = nrl_config.load_config(CFG)
    if OVERRIDES:
        cfg = nrl_config.parse_hydra_overrides(cfg, OVERRIDES)
    container = OmegaConf.to_container(cfg, resolve=True)
except Exception as e:  # noqa: BLE001
    print(f"  FAIL  load_config / overrides / resolve: {type(e).__name__}: {e}")
    sys.exit(1)
ok(f"load_config + {len(OVERRIDES)} launcher overrides + resolve")

try:
    from nemo_rl.algorithms.single_controller_utils.config import (  # noqa: E402
        MasterConfig,
        validate_single_controller_config,
    )
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc(limit=-6)
    print(f"  FAIL  cannot import NeMo-RL's SC config module: {type(e).__name__}: {e}")
    if STUBBED:
        print(f"  note  stubbed so far: {', '.join(sorted(STUBBED))}; run inside the container for the full check")
    sys.exit(2)
if STUBBED:
    print(f"  note  stubbed (not installed here): {', '.join(sorted(STUBBED))}")

try:
    mc = MasterConfig(**container)
    ok("MasterConfig(**config) (pydantic validators)")
except Exception as e:  # noqa: BLE001
    fail(f"MasterConfig: {e}")
    sys.exit(1)

# examples/run_grpo_single_controller.py, main()
if mc.grpo is None or mc.grpo.async_grpo is not None:
    fail(f"grpo.async_grpo must be null on SC (got {None if mc.grpo is None else mc.grpo.async_grpo})")
else:
    ok("grpo.async_grpo is null")
if not mc.data_plane.get("enabled", False):
    fail("data_plane.enabled must be true on SC")
else:
    ok("data_plane.enabled")

try:
    validate_single_controller_config(mc)
    ok("validate_single_controller_config")
except Exception as e:  # noqa: BLE001
    fail(f"validate_single_controller_config: {type(e).__name__}: {e}")

# setup_single_controller (single_controller_utils/setup.py), config-only parts
g = mc.grpo
if g.val_period > 0 or g.val_at_start or g.val_at_end:
    fail("SC has no validation: val_period 0, val_at_start/val_at_end false")
ck = mc.checkpointing
if ck["enabled"] and not ck.get("save_data_plane"):
    fail("checkpointing on with the in_order sampler needs checkpointing.save_data_plane: true")
if mc.data["use_multiple_dataloader"]:
    fail("data.use_multiple_dataloader must be false on SC")
gen = mc.policy["generation"]
use_gym = bool(mc.env.get("should_use_nemo_gym"))
if mc.token_capture.enabled:
    if not use_gym:
        fail("token_capture needs env.should_use_nemo_gym")
    if gen["backend"] != "vllm" or not gen["vllm_cfg"]["async_engine"]:
        fail("token_capture needs the vLLM backend with async_engine")
if gen["colocated"]["enabled"]:
    fail("SC needs non-colocated generation")

# ── expectations of the plan (feasibility report sections 1, 5, 6) ──────────────────────────────
a = mc.async_rl
p = mc.policy
cl = mc.cluster
vl = gen["vllm_cfg"]
gen_nodes = gen["colocated"]["resources"]["num_nodes"]
mg = p.get("megatron_cfg") or {}
dpc = mc.data_plane
prompts, gens = g.num_prompts_per_step, g.num_generations_per_prompt

def _get(obj, key):
    """Config sections are dicts (TypedDict) or pydantic models depending on the section."""
    if obj is None:
        return None
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


storage_units = _get(_get(dpc, "simple"), "num_storage_units")
print("  --- resolved")
print(f"  nodes: {cl['num_nodes']} x {cl['gpus_per_node']} GPUs = {cl['num_nodes'] - gen_nodes} trainer + {gen_nodes} vLLM")
print(f"  trainer: megatron {mg.get('enabled')} TP{mg.get('tensor_model_parallel_size')} EP{mg.get('expert_model_parallel_size')}, "
      f"cpu-offload optimizer {(mg.get('optimizer') or {}).get('optimizer_cpu_offload')}")
print(f"  vLLM: TP{vl['tensor_parallel_size']} EP{vl['expert_parallel_size']} (vLLM DP {vl['expert_parallel_size'] // vl['tensor_parallel_size']}), "
      f"async_engine {vl['async_engine']}, expose_http_server {vl.get('expose_http_server')}, max_model_len {vl.get('max_model_len')}")
print(f"  grpo: {prompts} prompts x {gens} = {prompts * gens}, train_global_batch_size {p['train_global_batch_size']}, "
      f"steps {g.max_num_steps}, epochs {g.max_num_epochs}, async_grpo {g.async_grpo}")
print(f"  length: max_total_sequence_length {p['max_total_sequence_length']}, max_new_tokens {gen['max_new_tokens']}, "
      f"packing {p['sequence_packing']['enabled']} ({p['sequence_packing'].get('train_mb_tokens')}/{p['sequence_packing'].get('logprob_mb_tokens')})")
print(f"  data_plane: enabled {_get(dpc, 'enabled')}, {_get(dpc, 'impl')}/{_get(dpc, 'backend')}, "
      f"num_storage_units {storage_units}")
print(f"  async_rl: sampler {a.sampler.name} lookahead {a.sampler.max_lookahead_versions}, "
      f"min_groups {a.min_groups_for_streaming_train}, inflight {a.max_inflight_prompts}, buffered {a.max_buffered_rollouts}, "
      f"recompute_kv {a.recompute_kv_cache_after_weight_updates}")
rf = a.rollout_failure
print(f"  rollout_failure: infra {rf.max_infra_attempts_per_prompt}, data {rf.max_data_attempts_per_prompt}, "
      f"skip {rf.max_skipped_prompts}, consecutive drops {rf.max_consecutive_dropped_prompts}, "
      f"nemo_gym.rollout_timeout_s {rf.nemo_gym.rollout_timeout_s}")
print(f"  stall_watchdog: {a.stall_watchdog.stall_timeout_s} s {a.stall_watchdog.stall_action}; "
      f"fleet_health {a.generation_fleet_health.enabled} (restart_dead_shards {a.generation_fleet_health.restart_dead_shards}, "
      f"refit_timeout_s {a.generation_fleet_health.refit_timeout_s})")
print(f"  token_capture: {mc.token_capture.enabled}; gym {use_gym}; validation {g.val_period}/{g.val_at_start}/{g.val_at_end}; "
      f"checkpointing {ck['enabled']}; loss IS correction {_get(mc.loss_fn, 'use_importance_sampling_correction')}")
print(f"  logger.log_dir {mc.logger['log_dir']}")


def expect(name, got, want):
    (ok if got == want else fail)(f"{name} = {got!r}" + ("" if got == want else f", expected {want!r}"))


print("  --- plan")
expect("cluster.num_nodes", cl["num_nodes"], 12)
expect("vLLM nodes", gen_nodes, 4)
expect("trainer EP", mg.get("expert_model_parallel_size"), 32)
expect("vLLM TP/EP", (vl["tensor_parallel_size"], vl["expert_parallel_size"]), (1, 4))
expect("vllm_cfg.async_engine", vl["async_engine"], True)
expect("grpo.max_num_steps", g.max_num_steps, 3)
expect("sampler", (a.sampler.name, a.sampler.max_lookahead_versions), ("in_order", 0))
expect("one chunk per step (min_groups_for_streaming_train)", a.min_groups_for_streaming_train, prompts)
expect("max_inflight_prompts", a.max_inflight_prompts, prompts)
expect("max_buffered_rollouts", a.max_buffered_rollouts, prompts)
expect("data_plane num_storage_units", storage_units, 2 * cl["num_nodes"])
expect("generation_fleet_health.enabled", a.generation_fleet_health.enabled, False)
expect("checkpointing.enabled", ck["enabled"], False)
expect("stall_action", a.stall_watchdog.stall_action, "abort")
if EXPECT == "a":
    expect("prompts x generations", (prompts, gens), (32, 16))
    expect("train_global_batch_size", p["train_global_batch_size"], 512)
    expect("max_total_sequence_length", p["max_total_sequence_length"], 2048)
    expect("env.should_use_nemo_gym", use_gym, False)
    expect("token_capture.enabled", mc.token_capture.enabled, False)
else:
    hcfg = mc.env["nemo_gym"]["harbor_agent"]["responses_api_agents"]["harbor_agent"]
    expect("prompts x generations", (prompts, gens), (8, 8))
    expect("train_global_batch_size", p["train_global_batch_size"], 64)
    expect("max_total_sequence_length", p["max_total_sequence_length"], 4096)
    expect("max_new_tokens", gen["max_new_tokens"], 1024)
    expect("env.should_use_nemo_gym", use_gym, True)
    expect("token_capture.enabled", mc.token_capture.enabled, True)
    expect("rollout_failure infra/data attempts", (rf.max_infra_attempts_per_prompt, rf.max_data_attempts_per_prompt), (2, 2))
    expect("rollout_timeout_s", rf.nemo_gym.rollout_timeout_s, 1800)
    expect("stall_timeout_s", a.stall_watchdog.stall_timeout_s, 2100)
    conc = hcfg["concurrency"]
    (ok if conc >= a.max_inflight_prompts * gens else fail)(
        f"harbor concurrency {conc} >= max_inflight_prompts x generations {a.max_inflight_prompts * gens}")
    print(f"  harbor: max_turns {hcfg['harbor_agent_kwargs']['max_turns']}, agent/verifier max timeout "
          f"{hcfg['harbor_agent_max_timeout']}/{hcfg['harbor_verifier_max_timeout']}, jobs {hcfg['harbor_jobs_dir']}")

if failures:
    print(f"  => {len(failures)} failure(s)")
    sys.exit(1)
print("  => config OK")
