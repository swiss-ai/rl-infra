"""Copy a Harbor jobs directory without the per-turn prompt token ids (and the agent trajectories), for download.

  python3 -I slim_harbor_jobs.py <harbor-jobs/RUN> [--keep-trajectories]
      writes <harbor-jobs/RUN>-slim, prints both sizes

Harbor 0.25's result.json keeps agent_result.rollout_details[*].prompt_token_ids: the whole prompt of every turn,
so it grows with the square of the turns (up to 151 MB per trial at 64K context, about 6 GB per 4 x 8 step). NeMo-RL
logs the trained tokens itself (train_data_step*.jsonl), and the trial analysis needs only rewards, exceptions and
timings. The agents' trajectory.json (the message history, ~10 MB per trial at 64K) is left out too unless
--keep-trajectories; fetch single ones from the original when needed (64K smoke: 6.18 GB -> 1.14 GB with them,
~0.13 GB without). Every other file and field is copied unchanged, with its permissions (result.json and
config.json hold the OpenSandbox API key: nothing is printed). A result.json that does not parse is copied as it
is. Delete the original on the cluster only after the slim copy is downloaded and analysed.
"""
import json
import os
import shutil
import sys


def slim_result(src, dst):
    """Write src without rollout_details' prompt_token_ids to dst; False if src is not a JSON object."""
    try:
        with open(src) as f:
            data = json.load(f)
    except ValueError:
        return False
    if not isinstance(data, dict):
        return False
    for detail in (data.get("agent_result") or {}).get("rollout_details") or []:
        if isinstance(detail, dict):
            detail.pop("prompt_token_ids", None)
    with open(dst, "w") as f:
        json.dump(data, f)
    shutil.copymode(src, dst)
    return True


def du(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def main():
    args = [a for a in sys.argv[1:] if a != "--keep-trajectories"]
    keep_trajectories = len(args) < len(sys.argv) - 1
    if len(args) != 1 or not os.path.isdir(args[0]):
        sys.exit("usage: python3 -I slim_harbor_jobs.py <harbor-jobs/RUN directory> [--keep-trajectories]")
    src = os.path.normpath(args[0])
    dst = src + "-slim"
    if os.path.exists(dst):
        sys.exit("%s exists already; remove it first" % dst)
    slimmed = copied = skipped = 0
    for root, _, files in os.walk(src):
        out = os.path.join(dst, os.path.relpath(root, src))
        os.makedirs(out, exist_ok=True)
        shutil.copymode(root, out)
        for name in files:
            if name == "trajectory.json" and not keep_trajectories:
                skipped += 1
                continue
            if name == "result.json" and slim_result(os.path.join(root, name), os.path.join(out, name)):
                slimmed += 1
                continue
            shutil.copy2(os.path.join(root, name), os.path.join(out, name))
            copied += 1
    print("%s: %.2f GB -> %s: %.2f GB (%d result.json slimmed, %d trajectory.json left out, %d files copied)"
          % (src, du(src) / 1e9, dst, du(dst) / 1e9, slimmed, skipped, copied))


if __name__ == "__main__":
    main()
