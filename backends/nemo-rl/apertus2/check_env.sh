#!/bin/bash
# Sanity check of the Apertus 2 stack: run it INSIDE the nemo-rl-apertus2 EDF on a GPU node
# (cwd /opt/nemo-rl) before the first job. Imports the modules the Ray actors import (trainer
# workers, Automodel setup, vLLM worker; Megatron-Bridge builds the Chonk provider from the
# checkpoint config), then builds Chonk on the meta device (no weights, no
# memory) and matches it against the checkpoint's shard headers with the patched loader's planner
# (per-expert keys stacked), so a silently dropped or missing weight shows up here.
# MODEL overrides the staged checkpoint.
set -u
MODEL="${MODEL:-/iopsstor/scratch/cscs/${USER}/models/apertus2-chonk-lc-iter3576}"
PY=/usr/bin/python3.12

echo "== python";   which "${PY}" uv ray; "${PY}" -V; echo "PYTHONPATH=${PYTHONPATH:-<unset>}"
echo "== env";      echo "NEMO_RL_PY_EXECUTABLES_SYSTEM=${NEMO_RL_PY_EXECUTABLES_SYSTEM:-<unset>} NRL_CONTAINER=${NRL_CONTAINER:-<unset>} PYDEPS=${PYDEPS:-<unset>}"
echo "== overlay";  bash "$(dirname "$0")/pydeps.sh" && cat "${PYDEPS}/.complete" && echo
echo "== imports";  "${PY}" -c "
import torch, transformers, vllm, fla, nemo_rl, nemo_automodel
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())
print('transformers', transformers.__version__, '| vllm', vllm.__version__, '| fla', getattr(fla, '__version__', '?'))
print('nemo_rl', nemo_rl.__file__); print('nemo_automodel', nemo_automodel.__file__)
# What the actors import (nemo_automodel itself is lazy and would hide a missing dependency).
import nemo_rl.models.automodel.setup, nemo_rl.models.policy.workers.dtensor_policy_worker_v2
import nemo_rl.models.generation.vllm.vllm_worker
from nemo_automodel.components.checkpoint.checkpointing import _load_hf_safetensors_sharded, plan_hf_safetensors_load
print('trainer + vLLM worker modules OK, Automodel loader patch present')
"
echo "== Gym (Harbor + OpenSandbox)"; "${PY}" -c "
import nemo_gym, nemo_rl.environments.nemo_gym as ng
print('nemo_gym', nemo_gym.__file__)
import nemo_gym.sandbox.providers.opensandbox.provider as p
print('opensandbox provider OK')
print('mask_degenerate_rollouts in nemo_rl:', 'mask_degenerate_rollouts' in open(ng.__file__).read())
import importlib.util as u
print('harbor_agent app importable spec:', u.find_spec('responses_api_agents.harbor_agent.app') is not None)
"
echo "== overlay imports"; "${PY}" -c "import soundfile, nemo.lens; print('soundfile', soundfile.__version__, '| nemo.lens', nemo.lens.__file__)"
echo "== Megatron trainer"; "${PY}" -c "
import megatron.core, megatron.bridge, deep_ep
print('megatron.core', megatron.core.__file__); print('megatron.bridge', megatron.bridge.__file__)
import nemo_rl.models.policy.workers.megatron_policy_worker
from megatron.bridge import AutoBridge
p = AutoBridge.from_hf_pretrained('${MODEL}', trust_remote_code=True).to_megatron_provider(load_weights=False)
print(type(p).__name__, '| layers', p.num_layers, '| experts', p.num_moe_experts, 'top', p.moe_router_topk, '| balancing', p.moe_router_load_balancing_type)
"
echo "== Chonk on meta vs checkpoint"; "${PY}" -c "
import json, os, torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from nemo_automodel.components.checkpoint.checkpointing import plan_hf_safetensors_load
M = '${MODEL}'
cfg = AutoConfig.from_pretrained(M, trust_remote_code=True)
with torch.device('meta'):
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
print(type(m).__name__, f'{sum(p.numel() for p in m.parameters())/1e9:.1f}B params')

# Checkpoint keys and shapes from the shard headers only (no tensor data is read).
wmap = json.load(open(os.path.join(M, 'model.safetensors.index.json')))['weight_map']
shapes = {}
for shard in sorted(set(wmap.values())):
    with safe_open(os.path.join(M, shard), framework='pt') as f:
        for k in f.keys():
            shapes[k] = tuple(f.get_slice(k).get_shape())

# The loader patch's own planner: which checkpoint keys feed which model tensor.
sd = m.state_dict()
plan, missing, unexpected = plan_hf_safetensors_load({k: tuple(v.shape) for k, v in sd.items()}, shapes)
bad = []
for name, src in plan.items():
    want = tuple(sd[name].shape)
    if src[0] == 'tensor':
        got = shapes[src[1]]
    elif src[2] == 'gate_up_proj':
        g = shapes[src[1] + '.0.gate_proj.weight']; got = (want[0], 2 * g[0], g[1])
    else:
        got = (want[0],) + shapes[src[1] + '.0.down_proj.weight']
    if got != want:
        bad.append((name, want, got))
print(f'model tensors {len(sd)} | planned {len(plan)}, of which grouped experts stacked from per-expert keys:', sum(s[0] == 'experts' for s in plan.values()))
print(f'missing from checkpoint, stay at init ({len(missing)}):', sorted({k.split(\".\")[-1] for k in missing}), missing[:3])
print(f'in checkpoint but not read, dropped ({len(unexpected)}):', unexpected[:5])
print(f'shape mismatches ({len(bad)}):', bad[:5])

tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)  # as NeMo-RL loads it
print('tokenizer', type(tok).__name__, 'bos', tok.bos_token, 'eos', tok.eos_token)
"
