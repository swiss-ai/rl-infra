"""ScienceWorld probe: JVM cost, latency, isolation of several simulators in one JVM, task inventory.

Run on a compute node (inside the NeMo-RL container, Python 3.12):
  PYTHONPATH=/home01/hanint06/sciworld/pydeps python3 -I sw_probe.py --java /home01/hanint06/sciworld/jre/bin/java \
      --out /home01/hanint06/sciworld/analysis/probe.json
"""
import argparse
import json
import os
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway

import scienceworld
from scienceworld.constants import BASEPATH, JAR_PATH

PY4J_JAR = os.path.join(
    os.path.dirname(os.path.dirname(scienceworld.__file__)), "py4j-0.10.9.9.data/data/share/py4j/py4j0.10.9.9.jar"
)
EASY = "teleportAction,openDoors,openContainers,selfWateringFlowerPots"


def rss_mb(pid: int) -> float:
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def cpu_s(pid: int) -> float:
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().split(")")[-1].split()
    ticks = os.sysconf("SC_CLK_TCK")
    return (int(parts[11]) + int(parts[12])) / ticks


class Jvm:
    def __init__(self, java: str, xmx: str, extra=()):
        opts = [f"-Xmx{xmx}", "-Xss4m", *extra]
        self.port, self.proc = launch_gateway(
            jarpath=PY4J_JAR, classpath=JAR_PATH, die_on_exit=True, cwd=BASEPATH, javaopts=opts,
            java_path=java, return_proc=True,
        )
        self.gw = JavaGateway(gateway_parameters=GatewayParameters(auto_field=True, port=self.port))

    def new(self):
        return self.gw.jvm.scienceworld.runtime.pythonapi.PythonInterface()

    def close(self):
        self.gw.shutdown()


def start(sim, task, var, simpl, gold=False):
    sim.load(task, var, simpl, gold)
    sim.reset()
    return sim.step("look around")


def dump(res, path):
    with open(path, "w") as f:
        json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--java", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xmx", default="4g")
    args = ap.parse_args()
    res = {"scienceworld_version": scienceworld.__version__ if hasattr(scienceworld, "__version__") else None}
    t0 = time.time()
    jvm = Jvm(args.java, args.xmx)
    res["jvm_start_s"] = time.time() - t0
    res["jvm_rss_mb_idle"] = rss_mb(jvm.proc.pid)
    sim = jvm.new()
    tasks = list(sim.getTaskNames())
    res["tasks"] = tasks
    print("tasks", len(tasks), flush=True)

    # 1. Inventory: variations and splits per task, task description, initial observation, gold path lengths.
    inv = {}
    for task in tasks:
        t = time.time()
        sim.load(task, 0, "", False)
        load_s = time.time() - t
        entry = {
            "max_variations": int(sim.getTaskMaxVariations(task)),
            "train": list(sim.getVariationsTrain()),
            "dev": list(sim.getVariationsDev()),
            "test": list(sim.getVariationsTest()),
            "load_s": load_s,
        }
        gold = {}
        for simpl in ("", EASY):
            lens = []
            for var in entry["train"][:4]:
                try:
                    sim.load(task, var, simpl, True)
                    lens.append(len(list(sim.getGoldActionSequence())))
                except Exception as e:  # noqa: BLE001
                    lens.append(f"error: {e}"[:200])
            gold[simpl or "none"] = lens
        entry["gold_len_train4"] = gold
        obs = start(sim, task, entry["train"][0], EASY)
        entry["task_desc_var0"] = sim.getTaskDescription()
        entry["first_obs_easy_var0"] = obs
        entry["first_obs_chars"] = len(obs)
        inv[task] = entry
        print(task, entry["max_variations"], len(entry["train"]), len(entry["dev"]), len(entry["test"]),
              {k: v for k, v in gold.items()}, round(load_s, 2), flush=True)
    res["inventory"] = inv
    dump(res, args.out)

    # 2. Latency of load+reset and of steps (random valid actions), one simulator.
    lat_load, lat_step, lat_valid = [], [], []
    rng = random.Random(0)
    for task in tasks[:: max(1, len(tasks) // 10)]:
        var = inv[task]["train"][1]
        t = time.time(); start(sim, task, var, EASY); lat_load.append(time.time() - t)
        for _ in range(15):
            t = time.time(); valid = list(sim.getValidActionObjectCombinations()); lat_valid.append(time.time() - t)
            act = rng.choice(valid) if valid else "look around"
            t = time.time(); sim.step(act); sim.getScore(); sim.getCompleted(); lat_step.append(time.time() - t)
        t = time.time(); sim.step("blorp the frobnicator"); lat_step.append(time.time() - t)
    res["invalid_action_reply"] = sim.step("blorp the frobnicator")
    res["latency"] = {
        "load_reset_look_s": {"mean": statistics.mean(lat_load), "max": max(lat_load)},
        "step_s": {"mean": statistics.mean(lat_step), "p95": sorted(lat_step)[int(0.95 * len(lat_step))], "max": max(lat_step)},
        "valid_actions_s": {"mean": statistics.mean(lat_valid), "max": max(lat_valid)},
    }
    print("latency", res["latency"], flush=True)
    # Warm load times per task (second load of another variation).
    warm = {}
    for task in tasks:
        t = time.time(); start(sim, task, inv[task]["train"][3 % len(inv[task]["train"])], EASY); warm[task] = time.time() - t
    res["warm_start_s"] = warm
    res["templates"] = {}
    for simpl in ("", EASY):
        start(sim, tasks[0], inv[tasks[0]]["train"][0], simpl)
        res["templates"][simpl or "none"] = list(sim.getPossibleActions())
    # Cross-talk of simplifications: A loaded with EASY, then B with none, then A tries to teleport.
    a, b = jvm.new(), jvm.new()
    start(a, tasks[0], inv[tasks[0]]["train"][0], EASY)
    start(b, tasks[0], inv[tasks[0]]["train"][1], "")
    res["crosstalk_teleport_after_other_load"] = a.step("teleport to kitchen")
    c = jvm.new(); start(c, tasks[0], inv[tasks[0]]["train"][0], EASY)
    res["teleport_alone"] = c.step("teleport to kitchen")
    print("crosstalk", repr(res["crosstalk_teleport_after_other_load"][:200]), "| alone", repr(res["teleport_alone"][:200]), flush=True)
    dump(res, args.out)

    # 3. Isolation: the same action script on one simulator alone, then on two interleaved in one JVM.
    def script(task):
        sim_ = jvm.new()
        start(sim_, task, inv[task]["train"][2], EASY)
        acts = []
        r = random.Random(1)
        for _ in range(12):
            valid = list(sim_.getValidActionObjectCombinations())
            acts.append(r.choice(valid))
            sim_.step(acts[-1])
        return acts

    ta, tb = tasks[0], tasks[5]
    acts_a, acts_b = script(ta), script(tb)

    def play(s, task, acts):
        out = [start(s, task, inv[task]["train"][2], EASY)]
        return out, [lambda a=a: (out.append(s.step(a)), out.append(str(s.getScore())))[0] for a in acts]

    a1 = jvm.new(); o1, f1 = play(a1, ta, acts_a)
    for f in f1: f()
    b1 = jvm.new(); p1, g1 = play(b1, tb, acts_b)
    for g in g1: g()
    a2, b2 = jvm.new(), jvm.new()
    o2, f2 = play(a2, ta, acts_a)
    p2, g2 = play(b2, tb, acts_b)
    for f, g in zip(f2, g2):
        f(); g()
    res["isolation_same_jvm_interleaved"] = (o1 == o2) and (p1 == p2)
    # Determinism across JVMs: start observation of a variation in a second JVM.
    jvm2 = Jvm(args.java, args.xmx)
    s2 = jvm2.new()
    res["determinism_across_jvms_first_obs"] = all(
        start(s2, t, inv[t]["train"][0], EASY) == inv[t]["first_obs_easy_var0"] for t in tasks
    )
    jvm2.close()
    print("isolation", res["isolation_same_jvm_interleaved"], "determinism", res["determinism_across_jvms_first_obs"], flush=True)

    # 4. Memory and throughput of N concurrent episodes in one JVM, driven by N threads.
    mem = []
    for n in (1, 16, 64, 128, 256):
        jv = Jvm(args.java, args.xmx)
        sims = [jv.new() for _ in range(n)]
        cpu0 = cpu_s(jv.proc.pid)
        lock = threading.Lock()

        def episode(i):
            r = random.Random(i)
            task = tasks[i % len(tasks)]
            with lock:
                start(sims[i], task, inv[task]["train"][i % len(inv[task]["train"])], EASY)
            steps = 0
            for _ in range(20):
                with lock:
                    valid = list(sims[i].getValidActionObjectCombinations())
                with lock:
                    sims[i].step(r.choice(valid) if valid else "look around")
                    sims[i].getScore(); sims[i].getCompleted()
                steps += 1
            return steps

        t = time.time()
        with ThreadPoolExecutor(max_workers=min(n, 64)) as ex:
            steps = sum(ex.map(episode, range(n)))
        wall = time.time() - t
        entry = {"n": n, "rss_mb": rss_mb(jv.proc.pid), "wall_s": wall, "steps": steps,
                 "jvm_cpu_s": cpu_s(jv.proc.pid) - cpu0, "threads": threading.active_count()}
        mem.append(entry)
        print("mem", entry, flush=True)
        res["episodes_per_jvm"] = mem
        dump(res, args.out)
        jv.close()
    res["episodes_per_jvm"] = mem
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", args.out, flush=True)
    jvm.close()


if __name__ == "__main__":
    main()
