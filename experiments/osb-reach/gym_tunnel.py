"""Gym's own per-sandbox model tunnel (Gym patch 0012, model_tunnel.py) on Breithorn, without a training
run: the exact code path of NemoGymSandboxEnvironment (AsyncSandbox with Gym's opensandbox.yaml, incl.
background_exec, one provider per sandbox, transport httpx as in the runs; ModelTunnel.open/close),
TUNNELS sandboxes in parallel, model_stub.py on this node as the tunnel target.

Per sandbox: start (python:3.12-slim, the micro-tasks' image), ModelTunnel.open() (upload chisel,
server start, node client, GET /health through the tunnel), then tunnel_probe.py in the pod against
127.0.0.1:8000 (short), ModelTunnel.close(), stop.

    python3 gym_tunnel.py
env: GYM_DIR, CHISEL_DIR, CHISEL_VERSION, TUNNELS (32), OLD_START (1: the fire-and-forget server start
     of job 5245999, to check whether it loses against the exec service), STREAM_S (5), IDLE_S (5)
"""

import asyncio
import logging
import os
import pathlib
import secrets
import statistics
import subprocess
import sys
import time

GYM_DIR = pathlib.Path(os.environ["GYM_DIR"])
sys.path.insert(0, str(GYM_DIR))

from omegaconf import OmegaConf  # noqa: E402

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, resolve_provider_config  # noqa: E402
from responses_api_agents.harbor_agent.custom_envs.nemo_gym_sandbox import model_tunnel as mt  # noqa: E402

HERE = pathlib.Path(__file__).parent
IMAGE = os.environ.get("REACH_IMAGE", "python:3.12-slim")
TUNNELS = int(os.environ.get("TUNNELS") or "32")
STREAM_S = int(os.environ.get("STREAM_S") or "5")
IDLE_S = int(os.environ.get("IDLE_S") or "5")

# force: importing nemo_gym already ran logging.debug() (nemo_gym/__init__.py _augment_sys_path), whose implicit
# basicConfig() installs a WARNING root handler; without force all INFO output is dropped (run 20261104-091416)
logging.basicConfig(
    level=logging.INFO, format="[gym_tunnel %(asctime)s] %(message)s", datefmt="%H:%M:%S", force=True
)
log = logging.getLogger("gym_tunnel")

if os.environ.get("OLD_START") == "1":
    # job 5245999's command: returns as soon as the server is backgrounded
    def _old_start(cfg, wait_s=15.0):
        b = mt.shlex.quote(cfg.pod_binary)
        return (
            f"chmod +x {b} && (setsid nohup {b} server --host 0.0.0.0 --port {cfg.server_port} --reverse "
            f"> {b}.log 2>&1 &)"
        )

    mt.server_start_script = _old_start
    log.info("OLD_START=1: fire-and-forget server start (as in job 5245999)")


async def one(provider_cfg: dict, idx: int, stub_port: int, metadata: dict) -> tuple[bool, float | None, str]:
    tag = f" [t{idx}]"
    cfg = mt.ModelTunnelConfig(
        chisel_dir=os.environ["CHISEL_DIR"],
        chisel_version=os.environ.get("CHISEL_VERSION", "1.12.0"),
        target=f"127.0.0.1:{stub_port}",
        tls_skip_verify=True,
    )
    spec = SandboxSpec(
        image=IMAGE,
        ttl_s=1800,
        metadata={**metadata, "purpose": "reach-test-gym-tunnel"},
        ports=(cfg.server_port,),
        provider_options={"secure_access": True},
    )
    sandbox = AsyncSandbox(resolve_provider_config(provider_cfg), spec)
    t0 = time.monotonic()
    await sandbox.start()
    log.info("%s sandbox up in %.1fs", tag, time.monotonic() - t0)
    tunnel = mt.ModelTunnel(sandbox, cfg, log, label=tag)
    try:
        t0 = time.monotonic()
        try:
            await tunnel.open()
        except Exception as e:  # noqa: BLE001
            log.info("%s OPEN FAILED: %s", tag, e)
            return False, None, "open"
        opened = time.monotonic() - t0
        await sandbox.upload(HERE / "tunnel_probe.py", "/tmp/tunnel_probe.py")
        res = await sandbox.exec(f"python3 /tmp/tunnel_probe.py {cfg.pod_port} {STREAM_S} {IDLE_S}", timeout_s=300)
        out = (res.stdout or "").strip().splitlines()
        ok = res.return_code == 0 and bool(out) and out[-1] == "PROBE OK"
        for line in out if not ok else out[-1:]:
            log.info("%s %s", tag, line)
        return ok, opened, "ok" if ok else "probe"
    finally:
        await tunnel.close()
        await sandbox.stop()


async def main() -> int:
    raw = OmegaConf.to_container(
        OmegaConf.load(GYM_DIR / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"), resolve=True
    )["sandbox"]
    # as the training runs (terminus-sc.yaml): the pooled aiohttp client breaks inside Ray workers
    raw["opensandbox"].setdefault("connection", {})["transport_backend"] = os.environ.get("TRANSPORT", "httpx")
    metadata = dict(raw.get("default_metadata", {}))
    log.info(
        "background_exec=%s, transport=%s, %d tunnels, image %s",
        raw["opensandbox"].get("operations", {}).get("background_exec"),
        raw["opensandbox"]["connection"]["transport_backend"],
        TUNNELS,
        IMAGE,
    )
    stub_port = 6000 + secrets.randbelow(1000)
    with open("model_stub.log", "wb") as fh:
        stub = subprocess.Popen([sys.executable, str(HERE / "model_stub.py"), str(stub_port)], stdout=fh, stderr=fh)
    try:
        await asyncio.sleep(1)
        results = await asyncio.gather(
            *(one(raw, i, stub_port, metadata) for i in range(TUNNELS)), return_exceptions=True
        )
    finally:
        stub.terminate()
        stub.wait()
    ok = [r for r in results if isinstance(r, tuple) and r[0]]
    failed_open = sum(1 for r in results if isinstance(r, tuple) and r[2] == "open")
    failed_probe = sum(1 for r in results if isinstance(r, tuple) and r[2] == "probe")
    errors = [r for r in results if isinstance(r, BaseException)]
    for e in errors:
        log.info("-- error: %r", e)
    times = sorted(r[1] for r in ok if r[1] is not None)
    if times:
        log.info(
            "-- open time min %.1fs median %.1fs max %.1fs", times[0], statistics.median(times), times[-1]
        )
    log.info(
        "-- RESULT: %d/%d tunnels OK (open failed %d, probe failed %d, other errors %d)%s",
        len(ok),
        TUNNELS,
        failed_open,
        failed_probe,
        len(errors),
        " [OLD_START]" if os.environ.get("OLD_START") == "1" else "",
    )
    return 0 if len(ok) == TUNNELS else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
