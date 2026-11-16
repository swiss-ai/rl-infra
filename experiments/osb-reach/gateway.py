"""Sandboxes through the ingress gateway (connection.use_server_proxy: false), as envs/harbor-verified-sft.yaml
sets since main's 2026-11-14 defaults, instead of through the OpenSandbox server's proxy. Same client code as the
runs: AsyncSandbox with Gym's opensandbox.yaml, transport httpx, background_exec, secure_access.

Per sandbox (GATEWAY_N in parallel): start; the execd endpoint's host (never its headers: they carry the
sandbox's access token); a short command; a LONG_S command (an idle timeout in the gateway would cut its status
polls); a 2 MB upload + download with a checksum; stop. Prints one line per failure and a RESULT line.

    python3 gateway.py
env: GYM_DIR, GATEWAY_N (16), LONG_S (75), GATEWAY_PROXY (0; 1 = through the server proxy, for comparison),
     REACH_IMAGE (python:3.12-slim), TRANSPORT (httpx)
"""

import asyncio
import collections
import hashlib
import logging
import os
import pathlib
import statistics
import sys
import tempfile
import time
from urllib.parse import urlsplit

GYM_DIR = pathlib.Path(os.environ["GYM_DIR"])
sys.path.insert(0, str(GYM_DIR))

from omegaconf import OmegaConf  # noqa: E402

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, resolve_provider_config  # noqa: E402

IMAGE = os.environ.get("REACH_IMAGE", "python:3.12-slim")
N = int(os.environ.get("GATEWAY_N") or "16")
LONG_S = int(os.environ.get("LONG_S") or "75")
PROXY = os.environ.get("GATEWAY_PROXY", "0") == "1"
EXECD_PORT = 44772  # opensandbox.constants.DEFAULT_EXECD_PORT; checked against the SDK below

# force: importing nemo_gym already installed a WARNING root handler (see gym_tunnel.py)
logging.basicConfig(level=logging.INFO, format="[gateway %(asctime)s] %(message)s", datefmt="%H:%M:%S", force=True)
log = logging.getLogger("gateway")


async def one(provider_cfg: dict, idx: int, metadata: dict, payload: pathlib.Path, digest: str, tmp: pathlib.Path):
    """Returns (failed phase or None, endpoint host, {phase: seconds}, error text)."""
    tag = f"[g{idx}]"
    spec = SandboxSpec(
        image=IMAGE,
        ttl_s=1800,
        metadata={**metadata, "purpose": "reach-test-gateway"},
        provider_options={"secure_access": True},
    )
    sandbox = AsyncSandbox(resolve_provider_config(provider_cfg), spec)
    times: dict[str, float] = {}
    host = "?"
    phase = "start"
    started = False
    try:
        t0 = time.monotonic()
        await sandbox.start()
        started = True
        times["start"] = time.monotonic() - t0

        # Informational only. Through the provider: AsyncSandbox.endpoint() answers only for ports declared in
        # SandboxSpec.ports, and the exec daemon's is not (job 5365400). Same lookup as the SDK's create, which
        # follows use_server_proxy; the readiness ping of start() already went to this endpoint.
        try:
            ep = await sandbox._provider.endpoint(sandbox._require_handle(), EXECD_PORT)
            host = urlsplit(ep.endpoint).netloc or ep.endpoint.split("/")[0]
        except Exception as e:  # noqa: BLE001
            host = f"lookup failed: {type(e).__name__}"

        phase = "exec"
        t0 = time.monotonic()
        res = await sandbox.exec("echo gateway-ok; uname -m", timeout_s=120)
        if res.return_code != 0 or "gateway-ok" not in (res.stdout or ""):
            raise RuntimeError(f"rc {res.return_code}, stdout {(res.stdout or '')[:200]!r}")
        times["exec"] = time.monotonic() - t0

        phase = "long_exec"
        t0 = time.monotonic()
        res = await sandbox.exec(f"sleep {LONG_S}; echo long-ok", timeout_s=LONG_S + 120)
        if res.return_code != 0 or "long-ok" not in (res.stdout or ""):
            raise RuntimeError(f"rc {res.return_code}, stdout {(res.stdout or '')[:200]!r}")
        times["long_exec"] = time.monotonic() - t0

        phase = "files"
        t0 = time.monotonic()
        await sandbox.upload(payload, "/tmp/gateway_payload.bin")
        back = tmp / f"back{idx}.bin"
        await sandbox.download("/tmp/gateway_payload.bin", back)
        if hashlib.sha256(back.read_bytes()).hexdigest() != digest:
            raise RuntimeError("downloaded file differs from the uploaded one")
        times["files"] = time.monotonic() - t0
        return None, host, times, ""
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:400]
        log.info("%s FAILED at %s after %s: %s", tag, phase, {k: round(v, 1) for k, v in times.items()}, err)
        return phase, host, times, err
    finally:
        if started:
            try:
                await sandbox.stop()
            except Exception as e:  # noqa: BLE001
                log.info("%s stop failed: %s: %s", tag, type(e).__name__, str(e)[:200])


async def main() -> int:
    try:
        from opensandbox.constants import DEFAULT_EXECD_PORT

        assert DEFAULT_EXECD_PORT == EXECD_PORT, DEFAULT_EXECD_PORT
    except ImportError:
        pass
    raw = OmegaConf.to_container(
        OmegaConf.load(GYM_DIR / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"), resolve=True
    )["sandbox"]
    conn = raw["opensandbox"].setdefault("connection", {})
    conn["transport_backend"] = os.environ.get("TRANSPORT", "httpx")
    conn["use_server_proxy"] = PROXY
    metadata = dict(raw.get("default_metadata", {}))
    log.info(
        "use_server_proxy=%s, transport=%s, background_exec=%s, %d sandboxes, LONG_S=%d, image %s",
        conn["use_server_proxy"],
        conn["transport_backend"],
        raw["opensandbox"].get("operations", {}).get("background_exec"),
        N,
        LONG_S,
        IMAGE,
    )
    with tempfile.TemporaryDirectory(dir=".") as d:
        tmp = pathlib.Path(d)
        payload = tmp / "payload.bin"
        payload.write_bytes(os.urandom(2 * 1024 * 1024))
        digest = hashlib.sha256(payload.read_bytes()).hexdigest()
        t0 = time.monotonic()
        results = await asyncio.gather(*(one(raw, i, metadata, payload, digest, tmp) for i in range(N)))
        wall = time.monotonic() - t0

    hosts = collections.Counter(r[1] for r in results)
    log.info("-- execd endpoint hosts: %s", dict(hosts))
    for ph in ("start", "exec", "long_exec", "files"):
        ts = sorted(r[2][ph] for r in results if ph in r[2])
        if ts:
            log.info("-- %-9s n=%d min %.1fs median %.1fs max %.1fs", ph, len(ts), ts[0], statistics.median(ts), ts[-1])
    failed = collections.Counter(r[0] for r in results if r[0])
    errs = collections.Counter(r[3].split(":")[0] for r in results if r[0])
    ok = N - sum(failed.values())
    log.info(
        "-- RESULT: %d/%d sandboxes OK through %s in %.0fs; failed by phase %s, by error %s",
        ok,
        N,
        "the server proxy" if PROXY else "the gateway",
        wall,
        dict(failed) or "{}",
        dict(errs) or "{}",
    )
    return 0 if ok == N else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
