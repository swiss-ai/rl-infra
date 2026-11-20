"""Build ScienceWorld rows for NeMo-RL (apertus2_sciworld_data_processor + env scienceworld).

For every task and split it starts the simulator at each variation it needs, records the task description and
the first observation (``look around``), and writes one prompt per row. Run on a compute node (container,
Python 3.12) with the scienceworld package on PYTHONPATH:

  python3 build_data.py --java .../jre/bin/java --out-dir .../data --simplifications <s> \
      --train-rows-per-task 200 --dev-per-task 5

Outputs, per action format F in {reply, tool}:
  sciworld_F_train.jsonl  task-balanced: each task gets --train-rows-per-task rows, cycling through its train
                          variations in a seeded random order (tasks with few variations repeat them)
  sciworld_F_dev.jsonl    the first --dev-per-task dev variations of each task, once each
  tasks.json              per task: topic, variation counts per split, the action templates
  load_check.json         rows whose first observation differs between load and load+reset
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time

TASK_INFO = None

# Command templates as the prompt lists them, keyed by the simulator's template name.
COMMANDS = [
    ("look around", "look around", "describe the room you are in"),
    ("look at OBJ", "look at <object>", "describe an object"),
    ("look in OBJ", "look in <object>", "describe a container's contents"),
    ("read OBJ", "read <object>", "read a note, book or label"),
    ("go OBJ", "go to <room>", "move to an adjacent room through an open door"),
    ("teleport OBJ", "teleport to <room>", "move to any room at once"),
    ("open OBJ", "open <object>", "open a door or container"),
    ("close OBJ", "close <object>", "close a door or container"),
    ("pick up OBJ", "pick up <object>", "put an object into your inventory"),
    ("put down OBJ", "put down <object>", "drop an object from your inventory"),
    ("move OBJ to OBJ", "move <object> to <object>", "put an object into or onto another"),
    ("pour OBJ in OBJ", "pour <object> in <object>", "pour a liquid or the contents of a container"),
    ("dunk OBJ in OBJ", "dunk <object> in <object>", "dip an object into a liquid"),
    ("mix OBJ", "mix <object>", "mix the contents of a container"),
    ("activate OBJ", "activate <object>", "switch a device on"),
    ("deactivate OBJ", "deactivate <object>", "switch a device off"),
    ("use OBJ on OBJ", "use <object> on <object>", "use a tool on an object, e.g. a thermometer"),
    ("connect OBJ to OBJ", "connect <object> to <object>", "connect two electrical terminals"),
    ("disconnect OBJ", "disconnect <object>", "disconnect an electrical component"),
    ("eat OBJ", "eat <object>", "eat something"),
    ("flush OBJ", "flush <object>", "flush a toilet"),
    ("focus on OBJ", "focus on <object>", "declare the object the task is about; see the notes"),
    ("wait", "wait", "let 10 time steps pass"),
    ("wait1", "wait1", "let 1 time step pass"),
    ("inventory", "inventory", "list what you carry"),
    ("task", "task", "repeat the task description"),
]

INTRO = (
    "You are an agent in ScienceWorld, a text simulation of a house and its surroundings with simple "
    "physics, chemistry and biology. You act by entering commands, one per turn."
)
NOTES = (
    "Write each command in plain words and name objects as the observations do, for example: look around, "
    "teleport to kitchen, open door to kitchen, look at picture, pick up thermometer, move metal pot to stove.\n\n"
    "Notes:\n"
    "- \"focus on <object>\" commits you to an object. Focus only on the objects the task tells you to focus on: "
    "focusing on any other object fails the task.\n"
    "- Time passes only through commands; processes such as heating or plant growth may need several "
    "\"wait\" commands.\n"
    "- If a command could mean several objects, the simulator lists them with numbers; answer with the "
    "number of the one you mean."
)
HOW = {
    "reply": (
        "Answer every turn with exactly one command as your reply and nothing else, for example: look around. "
        "The simulator's answer comes back as the next message."
    ),
    "tool": (
        "Enter each command with the act tool, one call per turn; its result is the simulator's answer. "
        "The episode ends when the task is complete or failed, or when you reply without calling the tool."
    ),
}


def prompt(fmt, task_desc, templates, first_obs):
    commands = "\n".join(
        f"- {shown} ({what})" for name, shown, what in COMMANDS if name in templates
    )
    return (
        f"{INTRO}\n\nYour task: {task_desc}\n\nCommands:\n{commands}\n\n{NOTES}\n\n{HOW[fmt]}\n\n"
        f"Observation:\n{first_obs}"
    )


def obs_hash(text):
    # Same as _obs_hash in nemo_rl/environments/scienceworld_environment.py: words in sorted order, because the
    # simulator's object order varies between JVMs.
    import re
    words = sorted(re.sub(r"[,.:()]", " ", text).split())
    return hashlib.sha1(" ".join(words).encode("utf-8")).hexdigest()[:16]


def worker(args):
    java, pkg, simpl, jobs = args
    sys.path.insert(0, pkg)
    import glob

    from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway
    from scienceworld.constants import BASEPATH, JAR_PATH

    jar = sorted(glob.glob(os.path.join(pkg, "py4j-*.data/data/share/py4j/py4j*.jar")))[-1]
    port = launch_gateway(jarpath=jar, classpath=JAR_PATH, die_on_exit=True, cwd=BASEPATH, java_path=java,
                          javaopts=["-Xmx2g", "-XX:+UseSerialGC", "-XX:ActiveProcessorCount=2", "-XX:-UsePerfData", "-Xlog:all=warning:stderr"])
    gw = JavaGateway(gateway_parameters=GatewayParameters(auto_field=True, port=port))
    sim = gw.jvm.scienceworld.runtime.pythonapi.PythonInterface()
    out = []
    for task, var in jobs:
        t = time.time()
        sim.load(task, var, simpl, False)
        first_load = sim.step("look around")
        desc = sim.getTaskDescription()
        load_s = time.time() - t
        sim.load(task, var, simpl, False)
        sim.reset()
        first_reset = sim.step("look around")
        # Rows use the standard API (load, then reset), as the environment's workers do; load alone gives
        # another first observation for about 13% of variations.
        out.append({"task": task, "variation": var, "task_desc": desc, "first_obs": first_reset,
                    "first_obs_load_only": first_load, "same_after_reset": first_load == first_reset,
                    "load_s": load_s})
    gw.shutdown()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--java", required=True)
    ap.add_argument("--package", default=os.path.dirname(os.path.dirname(os.path.abspath(__import__("scienceworld").__file__))))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--simplifications", required=True)
    ap.add_argument("--train-rows-per-task", type=int, required=True)
    ap.add_argument("--dev-per-task", type=int, required=True)
    ap.add_argument("--sample-per-task", type=int, default=4)
    ap.add_argument("--tasks", default="", help="comma-separated task names (default: all 30)")
    ap.add_argument("--processes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import scienceworld
    from scienceworld.constants import ID2TASK

    tasks_meta = json.load(open(os.path.join(os.path.dirname(scienceworld.__file__), "tasks.json")))
    meta_by_name = {m["task_name"]: m for m in tasks_meta}
    # Splits and templates from one simulator.
    sys.path.insert(0, args.package)
    import glob

    from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway
    from scienceworld.constants import BASEPATH, JAR_PATH

    jar = sorted(glob.glob(os.path.join(args.package, "py4j-*.data/data/share/py4j/py4j*.jar")))[-1]
    port = launch_gateway(jarpath=jar, classpath=JAR_PATH, die_on_exit=True, cwd=BASEPATH, java_path=args.java,
                          javaopts=["-Xmx2g", "-XX:+UseSerialGC", "-XX:ActiveProcessorCount=2", "-XX:-UsePerfData", "-Xlog:all=warning:stderr"])
    gw = JavaGateway(gateway_parameters=GatewayParameters(auto_field=True, port=port))
    sim = gw.jvm.scienceworld.runtime.pythonapi.PythonInterface()
    tasks = {}
    rng = random.Random(args.seed)
    keep = [t for t in args.tasks.split(",") if t]
    for task in sim.getTaskNames():
        if keep and task not in keep:
            continue
        sim.load(task, 0, args.simplifications, False)
        train = list(sim.getVariationsTrain())
        dev = list(sim.getVariationsDev())
        sim.step("look around")
        templates = list(sim.getPossibleActions())
        order = train[:]
        rng.shuffle(order)
        tasks[task] = {**meta_by_name.get(task, {}), "train": len(train), "dev": len(dev),
                       "test": len(list(sim.getVariationsTest())), "templates": templates,
                       "train_order": order, "dev_used": dev[: args.dev_per_task]}
        if task == "boil":
            valid = list(sim.getValidActionObjectCombinations())
            print("sample valid actions:", valid[:40], flush=True)
    gw.shutdown()

    needed = sorted({(t, v) for t, d in tasks.items()
                     for v in d["train_order"][: args.train_rows_per_task] + d["dev_used"]})
    print(f"{len(needed)} variations to start", flush=True)
    chunks = [needed[i:: args.processes] for i in range(args.processes)]
    t = time.time()
    with mp.get_context("spawn").Pool(args.processes) as pool:
        results = [r for part in pool.map(worker, [(args.java, args.package, args.simplifications, c) for c in chunks])
                   for r in part]
    print(f"started {len(results)} variations in {time.time() - t:.0f} s", flush=True)
    by_key = {(r["task"], r["variation"]): r for r in results}
    mismatch = [k for k, r in by_key.items() if not r["same_after_reset"]]
    json.dump({"load_vs_reset_mismatch": mismatch, "n": len(by_key),
               "examples": [[by_key[k]["first_obs_load_only"], by_key[k]["first_obs"]] for k in mismatch[:3]],
               "slowest_load_s": sorted(((r["load_s"], k) for k, r in by_key.items()), reverse=True)[:20]},
              open(os.path.join(args.out_dir, "load_check.json"), "w"), indent=1)
    print("load vs load+reset mismatches:", len(mismatch), flush=True)

    for fmt in ("reply", "tool"):
        for split in ("train", "dev", "sample"):
            rows = []
            for task, d in tasks.items():
                if split == "train":
                    order = d["train_order"]
                    variations = [order[i % len(order)] for i in range(args.train_rows_per_task)]
                elif split == "sample":
                    variations = d["train_order"][: args.sample_per_task]
                else:
                    variations = d["dev_used"]
                for v in variations:
                    r = by_key[(task, v)]
                    rows.append({
                        "messages": [
                            {"role": "user", "content": prompt(fmt, r["task_desc"], d["templates"], r["first_obs"])},
                            {"role": "assistant", "content": ""},
                        ],
                        "sciworld": {
                            "task": task, "variation": int(v), "simplifications": args.simplifications,
                            "action_format": fmt, "split": "dev" if split == "dev" else "train",
                            "topic": d.get("topic", ""),
                            "first_obs_hash": obs_hash(r["first_obs"]),
                        },
                    })
            if split == "train":
                random.Random(args.seed + 1).shuffle(rows)
            path = os.path.join(args.out_dir, f"sciworld_{fmt}_{split}.jsonl")
            with open(path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"wrote {len(rows)} rows to {path}", flush=True)
    for d in tasks.values():
        d.pop("train_order")
    json.dump(tasks, open(os.path.join(args.out_dir, "tasks.json"), "w"), indent=1)
    print("example prompt (reply):\n" + prompt("reply", "Your task is to boil water. ...", tasks["boil"]["templates"],
                                               "This room is called the hallway. ..."), flush=True)


if __name__ == "__main__":
    main()
