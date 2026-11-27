"""Runs INSIDE the sandbox (python:3.12-slim has no curl/nc): reports the pod's own network view, then
tries a TCP connection to every target and classifies the outcome. Standard library only.

    python3 probe.py host:port [host:port ...]
"""

import socket
import sys
import time
import urllib.request


def section(title: str) -> None:
    print(f"--- {title}", flush=True)


def probe(target: str) -> None:
    host, _, port = target.rpartition(":")
    t0 = time.monotonic()
    try:
        infos = socket.getaddrinfo(host, int(port), socket.AF_INET, socket.SOCK_STREAM)
    except socket.gaierror as e:
        print(f"{target:40s} DNS_FAIL     {e}", flush=True)
        return
    addr = infos[0][4]
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(8)
    try:
        s.connect(addr)
        t_conn = time.monotonic() - t0
        s.sendall(f"ping from {socket.gethostname()}\n".encode())
        try:
            reply = s.recv(256)
        except OSError as e:
            # Accepted but silent: a listener that does not answer, or a middlebox/proxy that accepts
            # every connection (would explain an agent hanging without an error).
            print(f"{target:40s} ACCEPTED_NO_REPLY {addr[0]} connect {t_conn:.2f}s, recv: {e}", flush=True)
            return
        print(f"{target:40s} CONNECTED    {addr[0]} connect {t_conn:.2f}s, reply {reply[:80]!r}", flush=True)
    except socket.timeout:
        print(f"{target:40s} TIMEOUT      {addr[0]} after {time.monotonic() - t0:.2f}s (dropped silently)", flush=True)
    except ConnectionRefusedError:
        print(f"{target:40s} REFUSED      {addr[0]} after {time.monotonic() - t0:.2f}s (host reached, port closed)", flush=True)
    except OSError as e:
        print(f"{target:40s} ERROR        {addr[0]} {e!r}", flush=True)
    finally:
        s.close()


def main() -> None:
    section("pod")
    print("hostname", socket.gethostname())
    try:
        print("addresses", sorted({i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}))
    except OSError as e:
        print("addresses", e)
    try:
        print(open("/etc/resolv.conf").read().strip())
    except OSError as e:
        print("resolv.conf", e)
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
        import os

        if os.environ.get(var):
            print(f"{var}={os.environ[var]}")
    section("public egress address")
    try:
        print(urllib.request.urlopen("https://api.ipify.org", timeout=10).read().decode())
    except Exception as e:  # noqa: BLE001 - report any failure
        print("failed:", repr(e))
    section("targets")
    for target in sys.argv[1:]:
        probe(target)


if __name__ == "__main__":
    main()
