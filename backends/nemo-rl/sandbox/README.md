# Code sandbox

Runs model-written code on a node of its own, for Gym's `code_gen` server. Without it, that code runs
under your account on any node of the job, with `/scratch` mounted.

A submission runs under [bubblewrap](https://github.com/containers/bubblewrap) with only `/usr`, the
system libraries and a read-only `/sandbox` mounted, so `/scratch` and home do not exist inside. It has
no network, an empty environment, a private process tree, and limits on memory, processes, files and
time. Everything inside dies with the sandbox. A container would not do here: HANGANG's enroot config
mounts `/scratch` into every container.

```bash
GYM_DIR=$NEMO_RL_DIR/3rdparty/Gym-workspace/Gym bash setup.sh   # once, on the login node: only copies files
JOB_TIME=06:00:00 bash start.sh                                 # one node, as a job of its own
bash ../../../run.sh ../../../envs/code-gen-sft.yaml            # reads $SCRATCH/code-sandbox/endpoint.json
```

Give the sandbox more time than the run, and `scancel` it afterwards. The server refuses to start
unless a test sandbox is cut off, and `code_gen` fails at start-up if the endpoint file is missing.

`test_sandbox.py` attacks a sandbox and then checks the service. Run it as a job after any change to
`sandbox_command`, or on a new cluster.

This is process isolation on a shared kernel, not a virtual machine. A submission can also still fake
its own result if it sets out to, because the runner lives in the same process. It cannot harm anything
that way, but do not trust a sudden jump in reward without reading the code.
