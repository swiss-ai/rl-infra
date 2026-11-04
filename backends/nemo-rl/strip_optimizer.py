#!/usr/bin/env python3
"""Removes the optimizer state from Megatron checkpoints of finished runs. This cannot be undone.

A torch_dist checkpoint keeps weights and optimizer state interleaved in the same .distcp shards. This
punches holes in the shards where the optimizer state was and takes its keys out of the index, so
nothing is copied and each weight stays at its offset. A checkpoint goes from about 1.5 TB to 0.2 TB of
disk, though `ls` still shows each shard at nearly its old size: use `du`. Its weights still load, but
training cannot resume from it.

The index changes last, so a job that dies halfway can simply run again. Until then, the optimizer keys
of that checkpoint read as zeros. Anything but the expected layout stops the script before it changes
a file.

    python3 strip_optimizer.py [--dry-run] <step dir> [<step dir> ...]
"""

import argparse
import ctypes
import glob
import os
import pickle
from collections import defaultdict

WEIGHT_PREFIXES = ("decoder.", "embedding.", "output_layer.", "rng_state/")
# AdamW checkpoints use the first, Muon checkpoints the second.
OPTIMIZER_PREFIXES = ("chained_", "optimizer.")
WEIGHTS_DIR = "policy/weights/iter_0000000"
MARKER = "OPTIMIZER_REMOVED"
# Holes stay inside optimizer state by at least this alignment, so no filesystem rounding can reach a weight.
HOLE_ALIGNMENT = 1024**2
FALLOC_FL_KEEP_SIZE = 0x01
FALLOC_FL_PUNCH_HOLE = 0x02


def punch_hole(file_descriptor, start, end):
    """Free the blocks of a byte range, which then reads as zeros."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]
    mode = FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE
    if libc.fallocate(file_descriptor, mode, start, end - start) != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"fallocate cannot punch a hole: {os.strerror(error)}")


def plan(metadata):
    """Find the optimizer state in each shard.

    Args:
        metadata: The checkpoint's index.

    Returns:
        The optimizer's keys and chunks, and each shard's byte ranges of optimizer state and end of data.
    """
    dropped_keys = set()
    dropped_chunks = []
    chunks = defaultdict(list)
    for index, chunk in metadata.storage_data.items():
        is_optimizer = index.fqn.startswith(OPTIMIZER_PREFIXES)
        if is_optimizer:
            dropped_keys.add(index.fqn)
            dropped_chunks.append(index)
        elif not index.fqn.startswith(WEIGHT_PREFIXES):
            raise ValueError(f"Unknown key {index.fqn!r}: not weights and not optimizer state.")
        chunks[chunk.relative_path].append((chunk.offset, chunk.length, is_optimizer))

    ranges = {}
    data_ends = {}
    for path, shard_chunks in chunks.items():
        shard_chunks.sort()
        position = 0
        shard_ranges = []
        for offset, length, is_optimizer in shard_chunks:
            if offset < position:
                raise ValueError(f"{path} has overlapping chunks at byte {offset}.")
            if is_optimizer and shard_ranges and shard_ranges[-1][1] == offset:
                shard_ranges[-1][1] = offset + length
            elif is_optimizer:
                shard_ranges.append([offset, offset + length])
            position = offset + length
        ranges[path] = shard_ranges
        data_ends[path] = position
    return dropped_keys, dropped_chunks, ranges, data_ends


def strip(step_dir, *, dry_run):
    """Strip one checkpoint.

    Args:
        step_dir: The checkpoint's step directory.
        dry_run: Whether to only report what would happen.

    Returns:
        The number of bytes of optimizer state in the index.
    """
    # Only here: the login node's plain Python can still show --help.
    from torch.distributed.checkpoint.metadata import Metadata  # noqa: F401

    weights_dir = os.path.join(step_dir, WEIGHTS_DIR)
    index_path = os.path.join(weights_dir, ".metadata")
    with open(index_path, "rb") as index_file:
        metadata = pickle.load(index_file)
    dropped_keys, dropped_chunks, ranges, data_ends = plan(metadata)

    shards = {os.path.basename(path) for path in glob.glob(os.path.join(weights_dir, "__*_*.distcp"))}
    if set(ranges) - shards:
        raise ValueError(f"{step_dir} lacks shards its index names: {sorted(set(ranges) - shards)}")
    size = sum(end - start for shard_ranges in ranges.values() for start, end in shard_ranges)
    if dry_run:
        print(f"{step_dir}: {size / 1e9:.1f} GB of optimizer state in {sum(map(len, ranges.values()))} ranges")
        return size

    blocks_before = sum(os.stat(os.path.join(weights_dir, shard)).st_blocks for shard in shards)
    for shard, shard_ranges in ranges.items():
        path = os.path.join(weights_dir, shard)
        file_descriptor = os.open(path, os.O_RDWR)
        try:
            for start, end in shard_ranges:
                if end == data_ends[shard]:
                    # Optimizer state to the end of the data: no later weight can be near it.
                    os.ftruncate(file_descriptor, min(start, os.fstat(file_descriptor).st_size))
                    continue
                first = (start + HOLE_ALIGNMENT - 1) // HOLE_ALIGNMENT * HOLE_ALIGNMENT
                last = end // HOLE_ALIGNMENT * HOLE_ALIGNMENT
                if first < last:
                    punch_hole(file_descriptor, first, last)
        finally:
            os.close(file_descriptor)
    blocks_after = sum(os.stat(os.path.join(weights_dir, shard)).st_blocks for shard in shards)

    if dropped_keys:
        for key in dropped_keys:
            del metadata.state_dict_metadata[key]
        for index in dropped_chunks:
            del metadata.storage_data[index]
        temporary = index_path + ".tmp"
        with open(temporary, "wb") as index_file:
            pickle.dump(metadata, index_file)
            index_file.flush()
            os.fsync(index_file.fileno())
        os.replace(temporary, index_path)
    with open(os.path.join(step_dir, MARKER), "w") as marker:
        marker.write("Weights only: strip_optimizer.py removed the optimizer state, so this cannot resume.\n")
    freed = (blocks_before - blocks_after) * 512
    print(f"{step_dir}: freed {freed / 1e9:.1f} GB", flush=True)
    return freed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("step_dirs", nargs="+")
    args = parser.parse_args()

    # Every checkpoint is checked before the first one changes.
    total = sum(strip(step_dir, dry_run=True) for step_dir in args.step_dirs)
    print(f"{len(args.step_dirs)} checkpoints, {total / 1e12:.2f} TB of optimizer state", flush=True)
    if not args.dry_run:
        total = sum(strip(step_dir, dry_run=False) for step_dir in args.step_dirs)
        print(f"freed {total / 1e12:.2f} TB", flush=True)


if __name__ == "__main__":
    main()
