"""Node -> pod through the OpenSandbox server proxy: can the node open a long-lived WebSocket into a
sandbox (the transport a per-sandbox reverse tunnel such as chisel needs)?

Creates one sandbox as the Harbor runs do, starts ws_server.py in it, resolves the port's proxy URL
and auth headers with Gym's provider.endpoint(), then: plain GET, WebSocket echo, round-trip times,
a 1 MiB message, and holding the connection idle-ish for HOLD_S seconds. Deletes the sandbox.

    python3 ws.py
env: as reach.py; WS_PORT (8765), HOLD_S (60)
"""

import asyncio
import base64
import os
import pathlib
import sys
import time

import aiohttp
from omegaconf import OmegaConf

from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider

IMAGE = os.environ.get("REACH_IMAGE", "python:3.12-slim")
PORT = int(os.environ.get("WS_PORT", "8765"))
HOLD_S = int(os.environ.get("HOLD_S", "60"))


def say(msg: str) -> None:
    print(f"[ws {time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def tests(url: str, headers: dict[str, str]) -> bool:
    ok = True
    async with aiohttp.ClientSession() as session:
        # 1. plain HTTP through the proxy
        try:
            async with session.get(f"{url}/hello", headers=headers, ssl=False, timeout=aiohttp.ClientTimeout(total=30)) as r:
                say(f"HTTP GET  -> {r.status} {(await r.text())[:100]!r}")
                ok &= r.status == 200
        except Exception as e:  # noqa: BLE001
            say(f"HTTP GET  -> FAILED {e!r}")
            ok = False
        # 2. WebSocket through the proxy
        ws_url = url.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/ws"
        try:
            async with session.ws_connect(ws_url, headers=headers, ssl=False, timeout=aiohttp.ClientWSTimeout(ws_receive=30), autoping=True, heartbeat=None) as ws:
                say("WS upgrade -> accepted")
                rtts = []
                for i in range(20):
                    t0 = time.monotonic()
                    await ws.send_str(f"ping {i}")
                    msg = await ws.receive(timeout=30)
                    rtts.append(time.monotonic() - t0)
                    if msg.data != f"ping {i}":
                        say(f"WS echo   -> WRONG {msg.type} {str(msg.data)[:80]!r}")
                        ok = False
                        break
                else:
                    rtts.sort()
                    say(f"WS echo   -> 20/20, rtt median {rtts[10] * 1000:.1f} ms, max {rtts[-1] * 1000:.1f} ms")
                big = os.urandom(1 << 20)
                t0 = time.monotonic()
                await ws.send_bytes(big)
                msg = await ws.receive(timeout=60)
                same = msg.type == aiohttp.WSMsgType.BINARY and msg.data == big
                say(f"WS 1 MiB  -> {'echoed' if same else 'FAILED ' + str(msg.type)} in {time.monotonic() - t0:.2f}s")
                ok &= same
                # hold: one message every 20 s (a tunnel with sparse traffic), check it survives
                t_end = time.monotonic() + HOLD_S
                n = 0
                while time.monotonic() < t_end:
                    await asyncio.sleep(min(20, max(0.0, t_end - time.monotonic())))
                    await ws.send_str(f"hold {n}")
                    msg = await ws.receive(timeout=30)
                    if msg.data != f"hold {n}":
                        say(f"WS hold   -> broke after {n} messages: {msg.type} {msg.data!r}")
                        ok = False
                        break
                    n += 1
                else:
                    say(f"WS hold   -> alive for {HOLD_S}s ({n} sparse messages)")
                await ws.close()
        except Exception as e:  # noqa: BLE001
            say(f"WS        -> FAILED {e!r}")
            ok = False
    return ok


async def main() -> int:
    cfg_path = pathlib.Path(os.environ["GYM_DIR"]) / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)["sandbox"]
    provider = OpenSandboxProvider(**cfg["opensandbox"])
    spec = SandboxSpec(
        image=IMAGE,
        ttl_s=900,
        metadata={**cfg.get("default_metadata", {}), "purpose": "reach-test-ws"},
        provider_options={"secure_access": True},
    )
    say(f"creating sandbox ({IMAGE}) ...")
    handle = await provider.create(spec)
    say(f"sandbox {handle.sandbox_id} up")
    try:
        script = base64.b64encode((pathlib.Path(__file__).parent / "ws_server.py").read_bytes()).decode()
        start = (
            f"echo {script} | base64 -d > /tmp/ws_server.py && "
            f"(setsid nohup python3 /tmp/ws_server.py {PORT} > /tmp/ws_server.log 2>&1 &) && sleep 2 && "
            "cat /tmp/ws_server.log"
        )
        res = await provider.exec(handle, start, timeout_s=60)
        say(f"pod server start rc={res.return_code}: {(res.stdout or '').strip()} {(res.stderr or '').strip()}")
        ep = await provider.endpoint(handle, PORT)
        url = ep.endpoint.rstrip("/")
        say(f"endpoint {url}  (header names: {sorted(ep.headers)})")
        ok = await tests(url, dict(ep.headers))
        res = await provider.exec(handle, "cat /tmp/ws_server.log", timeout_s=30)
        print((res.stdout or "").rstrip(), flush=True)
        say("RESULT: " + ("node -> pod HTTP and WebSocket through the proxy WORK" if ok else "FAILED (see above)"))
        return 0 if ok else 1
    finally:
        await provider.close(handle)
        await provider.aclose()
        say(f"sandbox {handle.sandbox_id} deleted")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
