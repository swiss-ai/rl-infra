"""Creates one OpenSandbox sandbox exactly as the Harbor runs do (Gym's OpenSandboxProvider with Gym's
own provider config, image python:3.12-slim, secure_access), runs probe.py in it against the given
targets, prints the result and deletes the sandbox.

    python3 reach.py host:port [host:port ...]
env: OPENSANDBOX_DOMAIN, OPENSANDBOX_API_KEY, NEMO_GYM_RUN_ID (sandbox label), GYM_DIR (Gym checkout)
"""

import asyncio
import base64
import os
import pathlib
import sys

from omegaconf import OmegaConf

from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider

IMAGE = os.environ.get("REACH_IMAGE", "python:3.12-slim")


async def main(targets: list[str]) -> int:
    cfg_path = pathlib.Path(os.environ["GYM_DIR"]) / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)["sandbox"]
    provider = OpenSandboxProvider(**cfg["opensandbox"])
    spec = SandboxSpec(
        image=IMAGE,
        ttl_s=900,
        metadata={**cfg.get("default_metadata", {}), "purpose": "reach-test"},
        provider_options={"secure_access": True},
    )
    print(f"[reach] creating sandbox ({IMAGE}) ...", flush=True)
    handle = await provider.create(spec)
    print(f"[reach] sandbox {handle.sandbox_id} up", flush=True)
    try:
        script = base64.b64encode((pathlib.Path(__file__).parent / "probe.py").read_bytes()).decode()
        cmd = f"echo {script} | base64 -d > /tmp/probe.py && python3 /tmp/probe.py {' '.join(targets)}"
        res = await provider.exec(handle, cmd, timeout_s=300)
        print(res.stdout or "", flush=True)
        if res.stderr:
            print("[reach] stderr:", res.stderr, flush=True)
        print(f"[reach] probe exit code {res.return_code}", flush=True)
        return res.return_code
    finally:
        await provider.close(handle)
        await provider.aclose()
        print(f"[reach] sandbox {handle.sandbox_id} deleted", flush=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
