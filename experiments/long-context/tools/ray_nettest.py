"""How fast Ray moves an object between nodes: the driver puts one array into the object store of the head node,
and a task pinned to each other node gets it. Run by submit_ray_nettest.sh as the driver command of ray.sub.
The training batch reaches the trainer nodes the same way (worker_groups.run_all_workers_sharded_data: ray.put of
each shard, the workers get it)."""
import os
import socket
import time

import numpy as np
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

GB = float(os.environ.get("NETTEST_GB", "8"))
ray.init(address="auto")
me = ray.get_runtime_context().get_node_id()
nodes = [n for n in ray.nodes() if n["Alive"]]
for n in nodes:
    print(f"node {n['NodeManagerHostname']} address {n['NodeManagerAddress']}"
          f"{' (driver)' if n['NodeID'] == me else ''}", flush=True)


@ray.remote(num_cpus=1)
def pull(refs):
    t = time.time()
    a = ray.get(refs[0])
    return socket.gethostname(), a.nbytes, time.time() - t


x = np.ones(int(GB * 2**30) // 8, dtype=np.int64)
t = time.time()
ref = ray.put(x)
print(f"ray.put of {x.nbytes / 2**30:.1f} GiB: {time.time() - t:.1f} s", flush=True)
for n in nodes:
    if n["NodeID"] == me:
        continue
    task = pull.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)).remote([ref])
    host, nbytes, dt = ray.get(task)
    print(f"RESULT pull to {host}: {nbytes / 1e6 / dt:.0f} MB/s ({nbytes / 2**30:.1f} GiB in {dt:.1f} s)", flush=True)
