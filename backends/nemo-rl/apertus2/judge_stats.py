"""Summarize how full each judge replica was, from env.judge_pool.stats_path.

Usage: judge_stats.py <judge_stats.jsonl> [start seconds into the run] [end seconds into the run]
"""

import json
import sys
from collections import defaultdict


def main():
    path, *window = sys.argv[1:]
    start, end = (float(window[0]), float(window[1])) if len(window) == 2 else (0.0, float("inf"))
    replicas = defaultdict(list)
    with open(path) as lines:
        for line in map(json.loads, lines):
            replicas[line["replica"]].append(line)
    first_time = min(rows[0]["time"] for rows in replicas.values())

    print("replica  running  peak  waiting  %time_waiting  calls/s  queue_s  prefill_s  cache_hit  peak_kv")
    for replica, rows in sorted(replicas.items()):
        rows = [row for row in rows if start <= row["time"] - first_time <= end]
        busy = [row for row in rows if row["num_requests_running"] + row["num_requests_waiting"] > 0]
        if len(rows) < 2 or not busy:
            print(f"{replica:7}  idle")
            continue
        added = {name: rows[-1][name] - rows[0][name] for name in rows[0] if name not in ("time", "replica")}
        calls = max(added["request_success"], 1)
        queries = max(added["prefix_cache_queries"], 1)
        print(
            f"{replica:7}  {sum(row['num_requests_running'] for row in busy) / len(busy):7.1f}"
            f"  {max(row['num_requests_running'] for row in rows):4.0f}"
            f"  {sum(row['num_requests_waiting'] for row in busy) / len(busy):7.1f}"
            f"  {100 * sum(row['num_requests_waiting'] > 0 for row in busy) / len(busy):13.0f}"
            f"  {added['request_success'] / (rows[-1]['time'] - rows[0]['time']):7.1f}"
            f"  {added['request_queue_time_seconds'] / calls:7.3f}"
            f"  {added['request_prefill_time_seconds'] / calls:9.3f}"
            f"  {added['prefix_cache_hits'] / queries:9.2f}"
            f"  {max(row['kv_cache_usage_perc'] for row in rows):7.2f}"
        )


if __name__ == "__main__":
    main()
