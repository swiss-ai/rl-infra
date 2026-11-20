"""Per training step: when the first and the last trainer rank started train(), and when the last one ended.
python3 stagger.py <ray-driver.log> [ranks]"""
import re, sys
from datetime import datetime
ranks = int(sys.argv[2]) if len(sys.argv) > 2 else 64
pat = re.compile(r"rank=(\d+) hostname=(\S+)\] train (start|end) ts=(\S+?)Z")
ev = []
with open(sys.argv[1], errors="replace") as f:
    for line in f:
        if "train start ts=" in line or "train end ts=" in line:
            m = pat.search(line)
            if m:
                t = datetime.strptime(m.group(4)[:26], "%Y-%m-%dT%H:%M:%S.%f").timestamp()
                ev.append((t, m.group(3), int(m.group(1)), m.group(2)))
starts = sorted(e for e in ev if e[1] == "start")
ends = sorted(e for e in ev if e[1] == "end")
print("step  first_start(UTC)  last_start-first  nodes_started  train_after_last_start  first_start->last_end")
i = 0
step = 0
while i < len(starts):
    grp = starts[i:i + ranks]
    step += 1
    t0, t1 = grp[0][0], grp[-1][0]
    e = [x for x in ends if x[0] > t0][:ranks]
    te = e[-1][0] if len(e) == ranks else None
    nodes = len(set(x[3] for x in grp))
    print("%4d  %s  %8.0f s  %4d  %s  %s" % (step, datetime.utcfromtimestamp(t0).strftime("%H:%M:%S"), t1 - t0, nodes,
          ("%8.0f s" % (te - t1)) if te else "       -", ("%8.0f s" % (te - t0)) if te else "       -"))
    i += ranks
