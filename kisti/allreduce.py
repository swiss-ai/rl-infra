# NCCL all-reduce check: correctness (KISTI saw wrong inter-node results with LL128) and bus bandwidth.
# One process per GPU; launched by nccl_2node.sh.
import os
import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
x = torch.empty(256 * 1024 * 1024 // 4, device="cuda")  # 256 MB of fp32
for _ in range(3):
    x.fill_(rank + 1)
    dist.all_reduce(x)
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    x.fill_(rank + 1)
    dist.all_reduce(x)
torch.cuda.synchronize()
dt = (time.time() - start) / 10
expected = world * (world + 1) / 2
ok = bool((x == expected).all().item())
if rank == 0:
    busbw = 2 * (world - 1) / world * 0.25 / dt
    print(f"world={world} correct={ok} (got {x[0].item()}, expected {expected}) "
          f"256MB all-reduce {dt * 1e3:.1f} ms, busbw {busbw:.1f} GB/s", flush=True)
dist.destroy_process_group()
