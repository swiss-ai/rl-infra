# Can an OpenSandbox pod reach a CSCS compute node?

Prerequisite for agents that run *inside* the sandbox (OpenCode, Claude Code, ...): their model calls
start in the pod and must reach the Gym model server on the training node. Job 5236016 (OpenCode,
`harbor-capture-sc` with a dead `HARBOR_MODEL_BASE_URL`) gave no answer: OpenCode printed nothing
for 900 s. This test asks the network question directly, without an agent.

    cd ~/rl-infra && bash experiments/osb-reach/run.sh

What it does (one debug node, under 5 min):

1. `run.sh` (login node): OpenSandbox key from `~/.opensandbox_token`, then `srun` one node in the
   training image (`EDF_NAME`, as the runs).
2. `reach_node.sh` (compute node): starts `listen.py` (TCP listener, port 5000-5999 like
   `policy_model`, logs every connection), self-tests it from the node, then runs `reach.py`.
3. `reach.py`: creates one sandbox with Gym's `OpenSandboxProvider` and Gym's own provider config
   (`python:3.12-slim`, `secure_access`, as the Harbor runs), runs `probe.py` in it, deletes it
   (TTL 15 min as a backstop; label `nemo-gym.nvidia.com/run=osb-reach-<date>`).
4. `probe.py` (in the pod, standard library only): the pod's hostname, `resolv.conf`, proxy
   variables, public egress address; then a TCP connect to each target.

Targets: every global IPv4 address of the node and its FQDN on the listener port; `pypi.org:443`
(internet control, expected to work);
`EXTRA_TARGETS="host:port ..."` adds more, e.g. a login node where you run `nc -lk <port>` yourself.

| Probe result | Meaning |
|---|---|
| `CONNECTED ... reply b'pong from nid...'` and a `CONNECTION from <ip>` line in the listener log | pods reach the node directly; the log shows the source address the node sees |
| `TIMEOUT` | packets dropped silently (firewall); an agent would hang until its own timeout |
| `REFUSED` | host reached, nothing listening on the port (or a firewall that rejects) |
| `ACCEPTED_NO_REPLY` | something accepted the connection but did not answer (proxy / middlebox); would also make an agent hang |
| `DNS_FAIL` | the name does not resolve in the pod |

Output: printed and saved to `$SCRATCH/osb-reach/<run id>/` (`reach.log`, `listen.log`).
Env: `ACCOUNT` (infra01), `PARTITION` (debug), `REACH_PORT`, `EXTRA_TARGETS`, `OPENSANDBOX_DOMAIN`,
`REACH_IMAGE` (python:3.12-slim), `NEMO_GYM_VENV_DIR` (`$HARBOR_ROOT/venvs`, as the Harbor runs),
`REACH_PYTHON`. The sandbox client needs the OpenSandbox SDK and `tenacity`, which the image's
system Python lacks; `reach_node.sh` uses Gym's `harbor_agent` server venv
(`$NEMO_GYM_VENV_DIR/responses_api_agents/harbor_agent/.venv`), built by any earlier Harbor run. If the job is killed before the sandbox is deleted, reap it as in
`docs/harbor-osb-chonk-run.md` (Cleanup) with the run id printed at the start.

## Reverse direction: node -> pod through the OpenSandbox proxy (`ws.py`)

Pods cannot reach the nodes (jobs 5238624, 5238974: silent drop), so in-pod agents need a tunnel the
node opens *into* each sandbox (planned: chisel/wstunnel reverse forward, pod `127.0.0.1:8000` ->
`policy_model`). Its transport is the OpenSandbox server route `/sandboxes/{id}/proxy/{port}/...`,
which upstream proxies for HTTP and WebSocket (`server/opensandbox_server/api/proxy.py` in
alibaba/OpenSandbox). `ws.py` checks that on Breithorn, after the TCP probe in the same job:

1. creates a second sandbox, starts `ws_server.py` (stdlib HTTP + WebSocket echo) on `WS_PORT` (8765);
2. resolves the port's proxy URL and auth headers with Gym's `provider.endpoint()` (API key, and the
   secure-access token since `secure_access` is on);
3. plain GET; WebSocket upgrade; 20 echoes with round-trip times; a 1 MiB message; one message every
   20 s for `HOLD_S` (60) seconds, as a quiet tunnel would.

`RESULT: node -> pod HTTP and WebSocket through the proxy WORK` means the tunnel transport exists.
A failing WebSocket with a working GET points at the front proxy (Traefik) not passing upgrades, or
an older server. `REACH_WS=0` skips this part.
