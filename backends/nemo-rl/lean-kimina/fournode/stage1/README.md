# Four-node Chonk GRPO, stage 1: measure the primitives on one node

The design ("Chonk GRPO on Four Nodes", https://claude.ai/artifact/MPHmnA4gYqkxTU5r7gdK7s) trains Chonk v2 on
4 GH200 nodes. Every GPU generates and trains in turn, gradients are summed in fp32 on the Grace CPUs, and Adam runs on
the CPUs over master weights, m and v that are streamed from scratch while the GPUs generate. Before building it, this
stage measures the numbers the design assumed or never measured. Each job uses one node and no training code.

| Job | Measurement | The design's assumption | Go if |
|---|---|---|---|
| host | `c2c`: pinned host <-> GPU, 4 GPUs at once | 450 GB/s, 18 GB of gradients per GPU in 40 ms | GPU->host >= 200 GB/s per GPU |
| host | `gradsum`: bf16 gradients to the host, summed into fp32, overlapped | 0.18 s per micro-batch | <= 0.5 s per micro-batch |
| host | `adam`: AdamW on host memory, per socket (foreach, fused) | not limiting | node's 27.1 G params in <= 60 s |
| host | `io`: scratch write/read/mixed, 8/16/32 files, O_DIRECT, default and 4x16 MiB striping | 4.7 GB/s (measured checkpoint writes) | read+write >= 4.5 GB/s per node |
| host | `stream-adam`: a real optimizer step, chunks read from scratch -> Adam -> written back, 4 sockets | 650 GB per node per step in ~140 s | node's 27.1 G params in <= 310 s (the generation phase) |
| vLLM (KV bf16, fp8) | continuous admission at 64/128/256 requests per GPU, GRPO Lean prompts, 16k, T 1.0 | 1,780 tok/s per busy GPU today; +27 % with more concurrency and fp8 KV | >= 1,780 tok/s per GPU without KV preemptions |
| vLLM | sleep level 1/2, wake-up, reload_weights, GPU memory after each | level-2 sleep frees the GPU; refit in 1-2 s | sleep 2 frees most GPU memory |

`stream-adam` also checks the in-place update of its first chunk against a reference AdamW on copies (`verify.ok`), so a
fast but wrong pipeline can't pass.

## Run

```bash
# on Clariden, after copying this folder's files to $SCRATCH/lean/fournode-stage1/code/
bash $SCRATCH/lean/fournode-stage1/code/submit.sh              # prints the three sbatch commands
bash $SCRATCH/lean/fournode-stage1/code/submit.sh --yes        # submits them (1 node each, preemptable, requeued)
python3 $SCRATCH/lean/fournode-stage1/code/report.py $SCRATCH/lean/fournode-stage1/out --md $SCRATCH/lean/fournode-stage1/report.md
```

- **host_job.sbatch** (about 35 min, limit 1 h) runs in the NeMo-RL container (`nemo-rl-apertus2`, the torch of Megatron's CPU
  Adam). It runs one task per GPU and Grace socket (`--cpu-bind=map_ldom`), so memory is local to the socket by first touch.
- It writes test files only under `$STAGE1/out/<job>-host/data` (up to about 190 GB) and removes them at the end.
- A requeued host job skips the measurements that already have a `done.<name>` marker.
- **vllm_job.sbatch** (about 65 min, limit 1:20) starts vLLM as the sampler does (the `chonk-proofnet` container, DP4 with
  expert parallelism on one node, `runai_streamer`), but with fp8 weights, GPU share 0.88 like the GRPO run, and sleep mode on.
  - If that start fails, it tries again without sleep mode, then with bf16 weights. `server.txt` says which configuration ran.
  - `conc_client.py` (in the sampler's venv) renders prompts with `sampler.Apertus2Format`, exactly as the GRPO task does.
  - Each level gets 480 s of warm-up, then 420 s of measurement from vLLM's `/metrics`.

## Caveats

- **One node is not four.** On 1 node each GPU holds a quarter of the experts (about 27 GB in fp8), and the expert all-to-all
  stays inside the node. On 4 nodes, vLLM would hold a sixteenth per GPU and send all-to-all traffic over Slingshot.
  - So this run has less KV cache per GPU than 4 nodes would have, and faster expert communication. Stage 2 has to repeat the
    best level on 4 nodes.
- **fp8 here is vLLM's `--quantization fp8`, which quantizes every linear layer.** The GRPO run uses NeMo-RL's fp8 path with
  `self_attn` kept in bf16. Attention projections are a small share of Chonk's weights.
- **The vLLM build is the sampler's image, not NeMo-RL's.** `server.txt` records both versions.
- **The streamed step's gradient is synthetic**, and the state is the right size for a socket, not real Chonk tensors.
  Speed doesn't depend on the values.
- **Each level starts with a ramp:** new requests start together. The 480 s warm-up spreads out their ages before the window.
