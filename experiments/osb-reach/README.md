# Can an OpenSandbox pod reach a CSCS compute node?

Prerequisite for agents that run *inside* the sandbox (OpenCode, Claude Code, ...): their model calls
start in the pod and must reach the Gym model server on the training node. Job 5236016 (OpenCode,
`harbor-capture-sc` with a dead `HARBOR_MODEL_BASE_URL`) gave no answer: OpenCode printed nothing
for 900 s. This test asks the network question directly, without an agent.

    cd ~/rl-infra && bash experiments/osb-reach/run.sh                        # tunnel test only (default)
    cd ~/rl-infra && REACH_TESTS="tcp ws tunnel" bash experiments/osb-reach/run.sh

`REACH_TESTS` picks the parts: `tcp` (pod -> node, below), `ws` (node -> pod through the proxy),
`tunnel` (chisel reverse tunnel). Results so far: tcp fails (jobs 5238624, 5238974, 5244154: silent
drop), ws works (job 5244154).

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
an older server.

## The tunnel: node opens it, pod uses it (`tunnel.py`, `REACH_TESTS=tunnel`)

The connection direction and the request direction differ. The node opens the WebSocket into the pod
(the only direction that works); inside it, chisel's reverse forward makes the pod-side server listen
on `127.0.0.1:8000`, and every connection there is carried back to the node, whose client dials the
target. The in-pod agent is the HTTP client, as it will be with `policy_model`.

    agent / tunnel_probe.py (pod) -> 127.0.0.1:8000 -> chisel server --reverse (pod, :8765)
        <= WebSocket, opened by the node: wss://<osb>/v1/sandboxes/<id>/proxy/8765 =>
    chisel client (node) -> model_stub.py 127.0.0.1:<port> (node; stands in for policy_model)

1. `run.sh` (login node) downloads chisel `CHISEL_VERSION` (1.12.0, github.com/jpillora/chisel) for
   linux amd64 and arm64 into `$SCRATCH/chisel` (`backends/nemo-rl/apertus2/fetch-chisel.sh`), checked against pinned sha256 (from the
   release's `checksums.txt`).
2. `tunnel.py` (node) starts `model_stub.py` (stdlib HTTP: JSON answer, POST echo, chunked stream),
   then `TUNNELS` (4) in parallel, each: create a sandbox as the Harbor runs; `uname -m`; upload the
   matching chisel and `tunnel_probe.py` with `provider.upload_file` (the pod needs no internet);
   start `chisel server --reverse --auth tunnel:<random>` on `TUNNEL_PORT` (8765); start
   `chisel client --auth ... --header <endpoint() headers> --tls-skip-verify <proxy url>
   R:127.0.0.1:8000:127.0.0.1:<stub>` on the node; wait for `Connected`; run the probe in the pod;
   print probe output and both chisel logs; stop the client; delete the sandbox.
3. `tunnel_probe.py` (pod, stdlib) uses only `127.0.0.1:8000`: first request (shows the node
   hostname that answered and a `/ng-rollout/<id>/...` path), 20 sequential, 16 concurrent, 1 MiB
   POST echo, a `STREAM_S` (60) s chunked stream (max gap between 1 s chunks), `IDLE_S` (120) s
   without traffic and one more request. `PROBE OK` / `PROBE FAIL: <checks>`.

`RESULT: n/n reverse tunnels pod 127.0.0.1:8000 -> node WORK` is the go for wiring it into the Harbor
runs. Test shortcuts: `--tls-skip-verify` (as `ws.py`); the API key and secure-access token are in
the client's argv on the node. Rollout-length holds: `IDLE_S=1500 JOB_TIME=00:45:00`.

## Gym's tunnel code at training concurrency (`gym_tunnel.py`, `REACH_TESTS=gymtunnel`, default)

Job 5245999 (training with `TUNNEL=1`, Gym patch 0013) failed 32/32 tunnels: the node's chisel client
got `websocket: close 1011` from the OpenSandbox proxy, i.e. nothing listened on the server port in the
pod. Patch 0013 started the server fire-and-forget (`(setsid nohup ... &)`, exec returns at once);
`tunnel.py` above, which worked, waited 2 s after the start. Patch 0014 makes the start command return
only once chisel logs `Listening on` and reports the pod-side log on failure.

`gym_tunnel.py` runs Gym's own `ModelTunnel` exactly as `NemoGymSandboxEnvironment` does
(`AsyncSandbox` with Gym's `opensandbox.yaml`, `background_exec`, one provider per sandbox, transport
httpx), `TUNNELS` (32) sandboxes in parallel, `model_stub.py` as the target; per sandbox: open (upload,
server start, client, `GET /health` through the tunnel), a short `tunnel_probe.py`, close, stop.
Needs patches 0013+0014 in the NeMo-RL clone's Gym (`setup-gym.sh`, see `harbor-capture-sc/README.md`).

    bash experiments/osb-reach/run.sh                 # patch 0014 start: expect RESULT: 32/32
    OLD_START=1 bash experiments/osb-reach/run.sh     # 0013's fire-and-forget start, for comparison

If `OLD_START=1` fails and the default passes, the race with the exec service was the cause of 5245999.

## Why do pods stay Pending? (`pending.py`, no kubectl needed)

Job 5317993 (SWE-Lego): all 512 creates ended in `HTTP 504 KUBERNETES::POD_READY_TIMEOUT`, Pending after
600 s; the server deletes the sandbox when its ready wait fails. `pending.py` reads the state while it waits,
with the tenant API key only: one sandbox per case, `GET /v1/sandboxes/{id}` (state, reason, message) and the
Kubernetes events (`GET /v1/sandboxes/{id}/diagnostics/events`) every 5 s, then delete. Standard library only;
runs on the login node (or any node that reaches the API), at most WAIT_S (900 s) + a few seconds.

    cd ~/rl-infra && python3 experiments/osb-reach/pending.py      # task.toml default: $SCRATCH/osb/swe_lego_v05/tasks/adamchainz__time-machine-298

| Case | Image | Resources | |
|---|---|---|---|
| `small` | `python:3.12-slim` | 1 CPU, 1024 MiB, 2 GiB | as the micro tasks (they run) |
| `swe-small` | the task's SWE-Lego image | 1 CPU, 1024 MiB, 2 GiB | does the image start? |
| `swe-full` | the task's SWE-Lego image | 4 CPU, 4096 MiB, 10 GiB | as job 5317993 |

`swe-small` Running and `swe-full` Pending: scheduling or quota (the events say `FailedScheduling` /
`Insufficient ...`). Both Pending: the image (events `Pulling` / `ErrImagePull` / `ImagePullBackOff`).
`small` Pending too: the tenant or the cluster. `--cases` picks cases, `--task-toml` another task.
