# Chonk GRPO on GSM8K

These configs train Apertus 2 Chonk (the long-context checkpoint at iteration 3576) with
async GRPO on GSM8K, on the Apertus 2 NeMo-RL stack in `backends/nemo-rl/apertus2`. That
stack pins the swiss-ai NeMo-RL fork together with the PRs and patches these runs need, and
runs an image whose UCCL is rebuilt with the fix for the illegal memory accesses in the MoE
combine (uccl-project/uccl#1085). `run.sh` at the repository root launches them; see the
top-level README.

The four recipes in `recipes/` build on these configs and reproduce the comparison of vLLM
against Megatron inference and of AdamW against Muon. Each run takes 58 steps of 128 prompts
with 16 generations each, at most one step off-policy, and evaluates on the GSM8K test split
at steps 0, 20, 40 and 58:

```bash
RECIPE=recipes/apertus2-chonk-vllm-adamw.yaml      bash run.sh envs/gsm8k.yaml   # 8 + 4 nodes
RECIPE=recipes/apertus2-chonk-megatron-adamw.yaml  bash run.sh envs/gsm8k.yaml   # 8 + 4 nodes
RECIPE=recipes/apertus2-chonk-vllm-muon.yaml       bash run.sh envs/gsm8k.yaml   # 16 + 4 nodes
RECIPE=recipes/apertus2-chonk-megatron-muon.yaml   bash run.sh envs/gsm8k.yaml   # 16 + 4 nodes
```

vLLM runs one inference replica per node. A Megatron inference replica spans two nodes,
because one node runs out of memory, so the same four nodes hold two replicas instead of
four. Muon needs 16 trainer nodes and 2048-token micro-batches; with 4096-token ones the
trainer runs out of memory around step 21. All recipes save the final weights with optimizer
state. Such saves used to hang often, until the 10-minute NCCL timeout killed the run; the
Megatron patches now free host memory around saves and loads, and no save hung in the tests
on HANGANG. They have not been retested on Alps.

`trainer-small-rx-buffers.yaml` is an option, off by default: add it to a run's `defaults` to
shrink libfabric's receive buffers on the trainers. In a 16-node AdamW run it freed about
3.5 GB per trainer process at the same speed, but only two steps were tested. Smaller buffers
can trigger flow control under heavy traffic, so watch for `Flow control` warnings in the log
and for slower steps.

The config names in this directory come from the experiments that led to the recipes.
`stage_model.sh` copies the checkpoint from capstor to your scratch and adds the Hugging
Face modeling code; `run.sh` calls it before every run.
