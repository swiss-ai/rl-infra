"""Turn the downloaded Apertus SFT mixture (HF dataset swiss-ai/apertus-sft-mixture, parquet shards) into the
on-disk Arrow format posttraining loads with load_from_disk. Subsampling happens at training time
(dataset_args.debug_subsample.train, after a seeded shuffle), so this keeps every conversation.

Usage: python prepare_data.py <downloaded dataset folder> <output folder>
"""
import collections
import glob
import os
import sys

from datasets import load_dataset


def main(src, dst):
    if os.path.exists(os.path.join(dst, "dataset_info.json")):
        print(f"== already prepared: {dst}")
        return
    files = sorted(glob.glob(os.path.join(src, "data", "train-*.parquet")))
    if not files:
        raise SystemExit(f"no parquet shards under {src}/data")
    ds = load_dataset("parquet", data_files=files, split="train")
    print(f"== {len(ds)} conversations from {len(files)} shards")
    counts = collections.Counter(ds["dataset_source"])
    for name, n in counts.most_common():
        print(f"   {n:9d}  {name}")
    tmp = dst + ".tmp"
    ds.save_to_disk(tmp)
    os.rename(tmp, dst)
    print(f"== saved: {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
