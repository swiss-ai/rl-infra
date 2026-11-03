"""Per-sandbox reverse tunnel through the OpenSandbox server proxy, end to end, as an in-pod agent
would use it: the NODE opens the connection (chisel client -> wss://<osb>/v1/sandboxes/<id>/proxy/
<port>), the POD uses it (127.0.0.1:8000 inside the pod -> model_stub.py on the node).

Per tunnel (TUNNELS in parallel): create a sandbox as the Harbor runs do; upload the chisel binary
for the pod's architecture and tunnel_probe.py (provider.upload_file, no internet needed in the pod);
start `chisel server --reverse` in the pod; start `chisel client ... R:127.0.0.1:8000:127.0.0.1:
<stub>` on the node with the proxy URL and auth headers from provider.endpoint(); run tunnel_probe.py
in the pod; print its output and the chisel logs; stop the client; delete the sandbox.

    python3 tunnel.py
env: as reach.py; CHISEL_DIR (dir with chisel_<ver>_linux_{amd64,arm64}, from run.sh), CHISEL_VERSION,
     TUNNELS (4), TUNNEL_PORT (8765, chisel server port in the pod), STREAM_S (60), IDLE_S (120)
"""

import asyncio
import os
import pathlib
import platform
import secrets
import subprocess
import sys
import time

from omegaconf import OmegaConf

from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider

HERE = pathlib.Path(__file__).parent
IMAGE = os.environ.get("REACH_IMAGE", "python:3.12-slim")
CHISEL_DIR = pathlib.Path(os.environ["CHISEL_DIR"])
CHISEL_VERSION = os.environ.get("CHISEL_VERSION", "1.12.0")
TUNNELS = int(os.environ.get("TUNNELS", "4"))
TUNNEL_PORT = int(os.environ.get("TUNNEL_PORT", "8765"))
STREAM_S = int(os.environ.get("STREAM_S", "60"))
IDLE_S = int(os.environ.get("IDLE_S", "120"))
POD_END = 8000  # what HARBOR_MODEL_BASE_URL would point at: http://127.0.0.1:8000
ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}


def chisel_bin(arch: str) -> pathlib.Path:
    return CHISEL_DIR / f"chisel_{CHISEL_VERSION}_linux_{arch}"


def say(tag: str, msg: str) -> None:
    print(f"[tunnel {time.strftime('%H:%M:%S')}] {tag} {msg}", flush=True)


async def wait_connected(log: pathlib.Path, proc: asyncio.subprocess.Process, timeout: float = 60) -> bool:
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        text = log.read_text(errors="replace") if log.exists() else ""
        if "Connected (Latency" in text:
            return True
        if proc.returncode is not None:
            return False
        await asyncio.sleep(0.5)
    return False


async def one(provider: OpenSandboxProvider, cfg: dict, idx: int, stub_port: int) -> bool:
    tag = f"t{idx}"
    spec = SandboxSpec(
        image=IMAGE,
        ttl_s=900 + IDLE_S + STREAM_S,
        metadata={**cfg.get("default_metadata", {}), "purpose": "reach-test-tunnel"},
        provider_options={"secure_access": True},
    )
    t0 = time.monotonic()
    handle = await provider.create(spec)
    say(tag, f"sandbox {handle.sandbox_id} up in {time.monotonic() - t0:.1f}s")
    client: asyncio.subprocess.Process | None = None
    log = pathlib.Path(f"chisel_client_{idx}.log")
    ok = False
    try:
        res = await provider.exec(handle, "uname -m", timeout_s=30)
        pod_arch = ARCH.get((res.stdout or "").strip(), "")
        if not pod_arch or not chisel_bin(pod_arch).exists():
            say(tag, f"no chisel for pod arch {(res.stdout or '').strip()!r} in {CHISEL_DIR}")
            return False
        t0 = time.monotonic()
        await provider.upload_file(handle, chisel_bin(pod_arch), "/tmp/chisel")
        await provider.upload_file(handle, HERE / "tunnel_probe.py", "/tmp/tunnel_probe.py")
        say(tag, f"uploaded chisel ({pod_arch}, {chisel_bin(pod_arch).stat().st_size >> 20} MiB) + probe in {time.monotonic() - t0:.1f}s")
        password = secrets.token_hex(16)  # per sandbox; the proxy port is reachable by anyone with the API key
        start = (
            "chmod +x /tmp/chisel && "
            f"(setsid nohup /tmp/chisel server --host 0.0.0.0 --port {TUNNEL_PORT} --reverse --auth tunnel:{password} "
            "> /tmp/chisel.log 2>&1 &) && sleep 2 && cat /tmp/chisel.log"
        )
        res = await provider.exec(handle, start, timeout_s=60)
        say(tag, f"pod chisel server rc={res.return_code}: {(res.stdout or '').strip()} {(res.stderr or '').strip()}")

        ep = await provider.endpoint(handle, TUNNEL_PORT)
        url = ep.endpoint.rstrip("/")
        say(tag, f"endpoint {url} (header names {sorted(ep.headers)})")
        # NB: header values (API key, secure-access token) end up in the client's argv on this node.
        hdr_args = [a for k, v in ep.headers.items() for a in ("--header", f"{k}: {v}")]
        node_arch = ARCH[platform.machine()]
        with log.open("wb") as fh:
            client = await asyncio.create_subprocess_exec(
                str(chisel_bin(node_arch)), "client", "--auth", f"tunnel:{password}", "--tls-skip-verify",
                "--max-retry-count", "5", *hdr_args, url, f"R:127.0.0.1:{POD_END}:127.0.0.1:{stub_port}",
                stdout=fh, stderr=subprocess.STDOUT,
            )
        t0 = time.monotonic()
        connected = await wait_connected(log, client)
        say(tag, f"node chisel client {'CONNECTED' if connected else 'NOT connected'} after {time.monotonic() - t0:.1f}s")
        if connected:
            res = await provider.exec(
                handle, f"python3 /tmp/tunnel_probe.py {POD_END} {STREAM_S} {IDLE_S}", timeout_s=STREAM_S + IDLE_S + 300
            )
            for line in (res.stdout or "").rstrip().splitlines() + (res.stderr or "").rstrip().splitlines():
                say(tag, line)
            ok = res.return_code == 0 and "PROBE OK" in (res.stdout or "")
        res = await provider.exec(handle, "tail -n 15 /tmp/chisel.log", timeout_s=30)
        for line in (res.stdout or "").rstrip().splitlines():
            say(tag, f"pod chisel: {line}")
    except Exception as e:  # noqa: BLE001
        say(tag, f"FAILED {e!r}")
    finally:
        if client is not None and client.returncode is None:
            client.terminate()
            await client.wait()
        for line in (log.read_text(errors="replace").rstrip().splitlines() if log.exists() else [])[-15:]:
            say(tag, f"node chisel: {line}")
        await provider.close(handle)
        say(tag, f"sandbox {handle.sandbox_id} deleted -> {'OK' if ok else 'FAILED'}")
    return ok


async def main() -> int:
    cfg_path = pathlib.Path(os.environ["GYM_DIR"]) / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)["sandbox"]
    provider = OpenSandboxProvider(**cfg["opensandbox"])
    stub_port = 6000 + secrets.randbelow(1000)
    stub_log = pathlib.Path("model_stub.log").open("wb")
    stub = subprocess.Popen([sys.executable, str(HERE / "model_stub.py"), str(stub_port)], stdout=stub_log, stderr=subprocess.STDOUT)
    say("--", f"model stub on 127.0.0.1:{stub_port}; {TUNNELS} tunnel(s), stream {STREAM_S}s, idle {IDLE_S}s")
    try:
        await asyncio.sleep(1)
        results = await asyncio.gather(*(one(provider, cfg, i, stub_port) for i in range(TUNNELS)))
    finally:
        stub.terminate()
        stub.wait()
        stub_log.close()
        await provider.aclose()
    n_req = sum('"' in line for line in pathlib.Path("model_stub.log").read_text(errors="replace").splitlines())
    say("--", f"model stub saw {n_req} requests (model_stub.log)")
    n_ok = sum(results)
    say("--", f"RESULT: {n_ok}/{TUNNELS} reverse tunnels pod 127.0.0.1:{POD_END} -> node "
              + ("WORK" if n_ok == TUNNELS else "FAILED (see above)"))
    return 0 if n_ok == TUNNELS else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
