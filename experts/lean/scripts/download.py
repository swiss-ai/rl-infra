"""Fetch a large file over HTTP in 32 MB byte ranges with 64 connections, keeping finished bytes.
Parts from an earlier 64-range download (parts/<i>) are kept; only their missing tails are fetched.
    python3 download.py <url> <size> <parts dir> <out file>
"""
import os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor
url, size, parts, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
N, CHUNK = 64, 32 * 1024 * 1024
part = (size + N - 1) // N
jobs = []  # (dest file, start, end)
for i in range(N):
    s, e = i * part, min((i + 1) * part, size) - 1
    p = os.path.join(parts, str(i))
    have = os.path.getsize(p) if os.path.exists(p) else 0
    pos, k = s + have, 0
    while pos <= e:
        ce = min(pos + CHUNK - 1, e)
        jobs.append((p + ".c%05d" % k, pos, ce)); pos = ce + 1; k += 1
print("chunks to fetch", len(jobs), flush=True)
def get(j):
    f, s, e = j
    if os.path.exists(f) and os.path.getsize(f) == e - s + 1:
        return
    for _ in range(10):
        if subprocess.call(["curl", "-sSfL", "--max-time", "600", "-r", "%d-%d" % (s, e), "-o", f, url]) == 0 and os.path.getsize(f) == e - s + 1:
            return
    raise RuntimeError("failed %s" % f)
with ThreadPoolExecutor(64) as ex:
    for n, _ in enumerate(ex.map(get, jobs), 1):
        if n % 20 == 0: print("done", n, flush=True)
with open(out, "wb") as g:
    for i in range(N):
        p = os.path.join(parts, str(i))
        for name in [str(i)] + sorted(x for x in os.listdir(parts) if x.startswith(str(i) + ".c")):
            fp = os.path.join(parts, name)
            if os.path.exists(fp):
                with open(fp, "rb") as h:
                    while True:
                        b = h.read(64 << 20)
                        if not b: break
                        g.write(b)
assert os.path.getsize(out) == size, os.path.getsize(out)
print("ok", out)
