"""Runs INSIDE the sandbox (standard library only), like an in-pod agent would: talks to the model
server ONLY through the tunnel's pod end 127.0.0.1:<port> (chisel reverse forward to model_stub.py
on the node).

    python3 tunnel_probe.py <port> <stream_s> <idle_s>
Checks: first request (and who answered), 20 sequential requests, 16 concurrent, 1 MiB POST echo, a
<stream_s>-second chunked stream (max gap between chunks), then <idle_s> s without traffic and one
more request. Last line: "PROBE OK" or "PROBE FAIL: ...".
"""

import concurrent.futures
import http.client
import json
import os
import socket
import sys
import time

HOST = "127.0.0.1"


def say(msg: str) -> None:
    print(f"[probe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def request(port: int, method: str, path: str, body: bytes | None = None, timeout: float = 60) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers={"Content-Type": "application/octet-stream"} if body else {})
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def main() -> int:
    port, stream_s, idle_s = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    fails: list[str] = []

    # The reverse listener appears once the node's chisel client has connected.
    t_end = time.monotonic() + 60
    while True:
        try:
            socket.create_connection((HOST, port), timeout=2).close()
            break
        except OSError as e:
            if time.monotonic() > t_end:
                say(f"tunnel end {HOST}:{port} never opened: {e!r}")
                print("PROBE FAIL: no tunnel listener", flush=True)
                return 1
            time.sleep(1)
    say(f"tunnel end {HOST}:{port} open (pod {socket.gethostname()})")

    def check(name: str, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            say(f"{name:<10} -> FAILED {e!r}")
            fails.append(name)

    def first():
        t0 = time.monotonic()
        status, body = request(port, "GET", "/ng-rollout/probe-0/training-token-capture/v1/models")
        d = json.loads(body)
        say(f"first      -> {status} from {d['from']} path={d['path']} in {(time.monotonic() - t0) * 1000:.1f} ms")
        assert status == 200

    def sequential():
        rtts = []
        for i in range(20):
            t0 = time.monotonic()
            status, _ = request(port, "GET", f"/seq/{i}")
            assert status == 200, status
            rtts.append(time.monotonic() - t0)
        rtts.sort()
        say(f"sequential -> 20/20, median {rtts[10] * 1000:.1f} ms, max {rtts[-1] * 1000:.1f} ms")

    def parallel():
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(16) as ex:
            res = list(ex.map(lambda i: request(port, "GET", f"/conc/{i}")[0], range(16)))
        ok = sum(s == 200 for s in res)
        say(f"concurrent -> {ok}/16 in {time.monotonic() - t0:.2f}s")
        assert ok == 16

    def big():
        data = os.urandom(1 << 20)
        t0 = time.monotonic()
        status, body = request(port, "POST", "/echo", data)
        say(f"1 MiB echo -> {status} {'same' if body == data else 'DIFFERENT'} in {time.monotonic() - t0:.2f}s")
        assert status == 200 and body == data

    def stream():
        conn = http.client.HTTPConnection(HOST, port, timeout=30)
        try:
            conn.request("GET", f"/stream?n={stream_s}&dt=1")
            resp = conn.getresponse()
            n, last, max_gap = 0, time.monotonic(), 0.0
            while line := resp.readline():
                if line.startswith(b"data:"):
                    now = time.monotonic()
                    max_gap = max(max_gap, now - last)
                    last = now
                    n += 1
        finally:
            conn.close()
        say(f"stream     -> {n}/{stream_s} chunks, max gap {max_gap:.2f}s (sent every 1 s)")
        assert n == stream_s and max_gap < 5

    def idle():
        t_end = time.monotonic() + idle_s
        while (left := t_end - time.monotonic()) > 0:
            time.sleep(min(30, left))
            say(f"idle       .. {idle_s - max(0, int(t_end - time.monotonic()))}/{idle_s}s")  # keeps the exec stream alive
        status, body = request(port, "GET", "/after-idle")
        say(f"after idle -> {status} {json.loads(body)['path']}")
        assert status == 200

    for name, fn in [("first", first), ("sequential", sequential), ("concurrent", parallel), ("1mib", big),
                     ("stream", stream), ("idle", idle)]:
        check(name, fn)
    print("PROBE OK" if not fails else f"PROBE FAIL: {' '.join(fails)}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
