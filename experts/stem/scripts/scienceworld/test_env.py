"""CPU check of the ScienceWorld environment (no GPU, no model): plays gold paths through env.step in both action
formats, checks rewards sum to the final score, replay after a lost episode, invalid commands, observation encoding,
and the data processor. Run in the NeMo-RL container with PYTHONPATH=<clone>:<pydeps-apertus2>:<pydeps>.
"""
import json
import os
import sys
import time

import ray

sys.path.insert(0, "/home01/hanint06/sciworld/pydeps")

from nemo_rl.data.apertus2_format import Apertus2Call, Apertus2Output  # noqa: E402
from nemo_rl.environments.scienceworld_environment import (  # noqa: E402
    ScienceWorldEnvironment,
    parse_act_arguments,
    parse_reply_command,
)

MODEL = "/scratch/hanint06/models/chonk-lc-16k-v0.2-iter_0018315"
SIMPL = "teleportAction,openDoors,openContainers,selfWateringFlowerPots"
CFG = {
    "num_workers": 2, "java_path": "/home01/hanint06/sciworld/jre/bin/java",
    "package_path": "/home01/hanint06/sciworld/pydeps", "jvm_heap": "2g", "tokenizer_path": MODEL,
    "simplifications": SIMPL, "max_actions": 100, "max_calls_per_turn": 2, "idle_timeout_s": 1800,
    "call_timeout_s": 120, "placement_group_prefix": None, "max_observation_chars": None,
}


def out(reply=None, calls=(), truncated=False, error=None):
    return Apertus2Output(think="", claim=None, reply=reply, calls=list(calls), error=error, truncated=truncated)


def gold(task, var):
    from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway
    from scienceworld.constants import BASEPATH, JAR_PATH
    import glob, os
    jar = sorted(glob.glob("/home01/hanint06/sciworld/pydeps/py4j-*.data/data/share/py4j/py4j*.jar"))[-1]
    port = launch_gateway(jarpath=jar, classpath=JAR_PATH, die_on_exit=True, cwd=BASEPATH,
                          java_path=CFG["java_path"], javaopts=["-Xmx2g"])
    gw = JavaGateway(gateway_parameters=GatewayParameters(auto_field=True, port=port))
    sim = gw.jvm.scienceworld.runtime.pythonapi.PythonInterface()
    sim.load(task, var, SIMPL, True)
    seq = list(sim.getGoldActionSequence())
    gw.shutdown()
    return seq


def play(env, spec, outputs, stop_on_end=True):
    meta = {"sciworld": spec}
    log = []
    total, steps = 0.0, []
    for o in outputs:
        log = log + [{"role": "assistant", "content": "", "apertus2_output": o}]
        r = ray.get(env.step.remote([log], [meta]))
        total += float(r.rewards[0])
        meta = r.metadata[0]
        steps.append((r.observations[0], float(r.rewards[0]), bool(r.terminateds[0])))
        log.append({"role": "user", "content": r.observations[0]["content"]})
        if r.terminateds[0] and stop_on_end:
            break
    return total, meta, steps


def main():
    assert parse_reply_command("**Action:** `open door to kitchen`.") == "open door to kitchen"
    assert parse_reply_command("I will look.\nAction: look around") == "look around"
    assert parse_reply_command("  ") is None
    assert parse_act_arguments('{"action": "Look Around"}') == "look around"
    assert parse_act_arguments("not json") is None
    ray.init(num_cpus=32, include_dashboard=False)
    env = ScienceWorldEnvironment.remote(CFG)
    report = {}
    for task, var in [("find-animal", 0), ("boil", 0), ("chemistry-mix", 0)]:
        seq = gold(task, var)
        for fmt in ("reply", "tool"):
            spec = {"task": task, "variation": var, "simplifications": SIMPL, "action_format": fmt, "split": "train"}
            if fmt == "reply":
                outputs = [out(reply=a) for a in seq]
            else:
                outputs = [out(calls=[Apertus2Call("act", i, json.dumps({"action": a}))]) for i, a in enumerate(seq)]
            t = time.time()
            total, meta, steps = play(env, spec, outputs)
            report[f"{task}/{fmt}"] = {"gold_len": len(seq), "total_reward": total, "end": meta["episode"]["end"],
                                       "score": meta["episode"]["score"], "s": round(time.time() - t, 2),
                                       "metrics": meta["frame_metrics"]}
            print(task, fmt, report[f"{task}/{fmt}"], flush=True)
    # Encoding of an observation and of a result.
    spec = {"task": "boil", "variation": 0, "simplifications": SIMPL, "action_format": "reply", "split": "train"}
    _, meta, steps = play(env, spec, [out(reply="look around"), out(reply="blorp"), out(reply=None),
                                      out(reply="x", calls=[Apertus2Call("python", 0, "{}")]),
                                      out(reply="reset task"), out(truncated=True)])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    for o, r, d in steps:
        print(repr(tok.decode(o["token_ids"])[:300]), r, d)
    print("reply edge cases metrics:", meta["frame_metrics"], meta["episode"]["end"])
    spec = {"task": "boil", "variation": 0, "simplifications": SIMPL, "action_format": "tool", "split": "train"}
    _, meta, steps = play(env, spec, [
        out(calls=[Apertus2Call("act", 0, '{"action": "look around"}'), Apertus2Call("act", 1, '{"action": "inventory"}'),
                   Apertus2Call("act", 2, '{"action": "task"}')]),
        out(calls=[Apertus2Call("act", 3, "{oops")]),
        out(reply="I am done."),
    ])
    for o, r, d in steps:
        print(repr(tok.decode(o["token_ids"])[:400]), r, d)
    print("tool edge cases metrics:", meta["frame_metrics"], meta["episode"]["end"])
    # Replay: play half a gold path, drop the episode from its worker, continue.
    seq = gold("boil", 0)
    spec = {"task": "boil", "variation": 0, "simplifications": SIMPL, "action_format": "reply", "split": "train"}
    meta = {"sciworld": spec}
    total = 0.0
    for i, a in enumerate(seq):
        log = [{"role": "assistant", "content": "", "apertus2_output": out(reply=a)}]
        r = ray.get(env.step.remote([log], [meta]))
        total += float(r.rewards[0])
        meta = r.metadata[0]
        if i == len(seq) // 2:
            # Simulate a lost episode: close it on its worker.
            workers = ray.get(env.worker_stats.remote())
            print("worker stats mid-episode:", workers)
            env_workers = None
        if i == len(seq) // 2:
            meta = json.loads(json.dumps(meta))
            meta["episode"]["id"] = "replayed-" + meta["episode"]["id"]  # unknown id => replay from history
        if r.terminateds[0]:
            break
    print("replay test: total", total, "end", meta["episode"]["end"], "replays", meta["episode"]["replays"],
          "mismatches", meta["episode"]["replay_mismatches"])
    report["replay"] = {"total": total, "end": meta["episode"]["end"], "replays": meta["episode"]["replays"],
                        "mismatch": meta["episode"]["replay_mismatches"]}
    # Processor.
    from nemo_rl.data.interfaces import TaskDataSpec
    from nemo_rl.data.processors import apertus2_sciworld_data_processor
    row = json.loads(open("/home01/hanint06/sciworld/data/sciworld_tool_sample.jsonl").readline())
    row["task_name"] = "x"
    ts = TaskDataSpec(task_name="x", system_prompt_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../frames/apertus2-system-sciworld-act.json"))
    d = apertus2_sciworld_data_processor(row, ts, tok, 4096, 0)
    print("prompt tokens:", d["length"], repr(tok.decode(d["message_log"][0]["token_ids"])[:700]))
    print("REPORT", json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
