"""Index of the per-step rollout files of a STEM-mix run: for every row of train_data_step<N>.jsonl the training
step, the prompt group, the dataset the prompt comes from and the prompt's row in that dataset.

A run started before NeMo-RL commit 71f91324d writes rollout files without these columns (job 1299180). They follow
from the file itself: the step is in the file name, rows come in blocks of GROUP answers per prompt, and the prompt
is the first message, which is the `input` (or first message) of exactly one row of one dataset file. Prompts found
in none of the files are DAPO-Math, the only dataset not read from a file here.

  python3 rollout_index.py <run log dir>/exp_NNN [out dir]     (Python 3.6, light: fine on the data mover)

Writes <out dir>/step<N>.tsv (default <exp dir>/rollout_index) for every step that has no index yet, with columns
step, row, prompt_group, dataset, dataset_row, reward, length; and prints the groups per dataset of each new step.
"""
import collections
import glob
import hashlib
import json
import os
import re
import sys

DATA = os.environ.get("STEM_DATA", "/scratch/hanint06/data/stem-rl")   # the run's training files
DATASETS = [  # name as in the run's num_prompts_per_dataloader, file
    ("run3-numina_clean", DATA + "/run3/numina_clean.jsonl"),
    ("run3-mcqa_science_cs", DATA + "/run3/mcqa_science_cs.jsonl"),
    ("openqa-openqa_science", DATA + "/openqa/openqa_science.jsonl"),
    ("reasoning_gym-train", DATA + "/reasoning_gym/train.jsonl"),
    ("train-enigmata_v1", DATA + "/enigmata/train/enigmata_v1.jsonl"),
    ("scienceworld-sciworld_act_train", DATA + "/scienceworld/sciworld_act_train.jsonl"),
    ("chem_faithful-train", DATA + "/chem_faithful/train.jsonl"),
]
GROUP = 16


def key(text):
    return hashlib.md5(text.strip().encode("utf-8", "replace")).digest()


def prompt_table():
    table = {}
    for name, path in DATASETS:
        with open(path) as f:
            for row, line in enumerate(f):
                r = json.loads(line)
                text = r["input"] if "input" in r else r["messages"][0]["content"]
                table.setdefault(key(text), (name, row))  # a repeated prompt keeps its first row
    return table


def one(value):
    return value[0] if isinstance(value, list) and len(value) == 1 else value


def messages(content):
    content = one(content)
    return content if isinstance(content, list) else [content]


def main():
    exp = sys.argv[1].rstrip("/")
    out = sys.argv[2] if len(sys.argv) > 2 else exp + "/rollout_index"
    os.makedirs(out, exist_ok=True)
    todo = []
    for path in glob.glob(exp + "/train_data_step*.jsonl"):
        step = int(re.search(r"step(\d+)\.jsonl$", path).group(1))
        if not os.path.exists("%s/step%d.tsv" % (out, step)):
            todo.append((step, path))
    if not todo:
        print("nothing new in", exp)
        return
    table = prompt_table()
    for step, path in sorted(todo):
        groups = collections.Counter()
        tmp = "%s/step%d.tsv.tmp" % (out, step)
        with open(path) as f, open(tmp, "w") as g:
            g.write("step\trow\tprompt_group\tdataset\tdataset_row\treward\tlength\n")
            for row, line in enumerate(f):
                r = json.loads(line)
                name, dataset_row = "DAPOMath17K", ""
                for text in messages(r["content"])[:3]:
                    hit = table.get(key(text)) if isinstance(text, str) else None
                    if hit:
                        name, dataset_row = hit
                        break
                if row % GROUP == 0:
                    groups[name] += 1
                g.write("%d\t%d\t%d\t%s\t%s\t%s\t%s\n" % (
                    step, row, row // GROUP, name, dataset_row, one(r["rewards"]), one(r["input_lengths"])))
        os.rename(tmp, "%s/step%d.tsv" % (out, step))
        print("step %d: %d rows, groups per dataset: %s" % (
            step, row + 1, ", ".join("%s %d" % kv for kv in sorted(groups.items()))))


main()
