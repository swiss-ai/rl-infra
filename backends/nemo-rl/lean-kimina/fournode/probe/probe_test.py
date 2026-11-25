"""Laptop tests of the probe's scripts: python3 -m pytest -q probe_test.py"""
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report  # noqa: E402

RANKS = 16


# ---- probe_driver.sh with a stand-in NeMo-RL driver

STUB = r"""#!/bin/bash
# records its arguments, then "trains": creates step_<N> for N up to the last grpo.max_num_steps
printf '%s\n' "$*" >> "$CALLS"
steps=""; ckpt=""
for a in "$@"; do
  [[ "$a" == grpo.max_num_steps=* ]] && steps="${a#*=}"
  [[ "$a" == checkpointing.checkpoint_dir=* ]] && ckpt="${a#*=}"
done
[[ -n "${NO_CKPT:-}" ]] || for i in $(seq 1 "$steps"); do mkdir -p "$ckpt/step_$i"; done
exit "${STUB_RC:-0}"
"""


def run_driver(tmp_path, *extra_args, env=None, pre=None):
    stub = tmp_path / "stub.sh"
    stub.write_text(STUB)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "ray").write_text("#!/bin/bash\necho ' 0.0/16.0 GPU'\n")
    (bin_dir / "ray").chmod(0o755)
    calls = tmp_path / "calls.txt"
    log_dir, ckpt = tmp_path / "logs", tmp_path / "results"
    if pre:
        pre(ckpt)
    e = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", CALLS=str(calls), PROBE_DRIVER=f"bash {stub}",
             PROBE_STEPS="5", PROBE_FIRST_STEPS="3")
    e.update(env or {})
    args = ["examples/run_grpo.py", "--config", "c.yaml", "grpo.max_num_steps=200", f"logger.log_dir={log_dir}",
            f"checkpointing.checkpoint_dir={ckpt}", "grpo.max_num_steps=5", *extra_args]
    p = subprocess.run(["bash", str(HERE / "probe_driver.sh"), *args], env=e, capture_output=True, text=True)
    lines = calls.read_text().splitlines() if calls.exists() else []
    marks = (log_dir / "probe-phases.tsv").read_text().splitlines() if (log_dir / "probe-phases.tsv").exists() else []
    return p, lines, marks


def steps_in(call):
    return {a for a in call.split() if a.startswith("grpo.max_num_steps=")}


def test_two_phases_with_resume_and_new_deadline(tmp_path):
    end = int(time.time()) + 7200
    p, calls, marks = run_driver(tmp_path, "checkpointing.checkpoint_must_save_by=00:03:30:00",
                                 env={"NRL_JOB_END_EPOCH": str(end), "NRL_SAVE_MARGIN_S": "1800"})
    assert p.returncode == 0, p.stderr
    assert len(calls) == 2
    assert steps_in(calls[0]) == {"grpo.max_num_steps=3"} and steps_in(calls[1]) == {"grpo.max_num_steps=5"}
    assert "checkpoint_must_save_by=00:03:30:00" in calls[0]       # phase A keeps driver_retry.sh's value
    save_by = [a for a in calls[1].split() if a.startswith("checkpointing.checkpoint_must_save_by=")]
    assert len(save_by) == 1
    d, h, m, s = map(int, save_by[0].split("=")[1].split(":"))
    assert abs(d * 86400 + h * 3600 + m * 60 + s - 5400) < 60      # job end - now - margin
    assert [x.split("\t")[1:] for x in marks] == [["A", "start steps=3"], ["A", "end rc=0"],
                                                  ["B", "start steps=5"], ["B", "end rc=0"]]
    assert "phase B start" in p.stdout


def test_phase_a_skipped_when_its_checkpoint_exists(tmp_path):
    p, calls, marks = run_driver(tmp_path, pre=lambda c: (c / "step_3").mkdir(parents=True))
    assert p.returncode == 0 and len(calls) == 1 and steps_in(calls[0]) == {"grpo.max_num_steps=5"}
    assert marks[0].split("\t")[1:] == ["A", "skipped (step_3 exists)"]


def test_failed_phase_a_ends_the_probe(tmp_path):
    p, calls, marks = run_driver(tmp_path, env={"STUB_RC": "7"})
    assert p.returncode == 7 and len(calls) == 1 and marks[-1].endswith("end rc=7")


def test_phase_a_without_checkpoint_ends_the_probe(tmp_path):
    p, calls, _ = run_driver(tmp_path, env={"NO_CKPT": "1"})
    assert p.returncode == 3 and len(calls) == 1 and "without" in p.stderr


def test_shell_syntax():
    for name in ("probe.sh", "probe_driver.sh", "probe_watch.sh"):
        subprocess.run(["bash", "-n", str(HERE / name)], check=True)


# ---- probe.sh: the settings it hands the job

def test_probe_env_stays_off_shared_folders(tmp_path):
    scratch = tmp_path / "scratch"
    e = dict(os.environ, SCRATCH=str(scratch), HOME=str(tmp_path / "home"), RUN="r1", VLLM_UTIL="0.7",
             EXTRA_OVERRIDES="x=1")
    p = subprocess.run(["bash", str(HERE / "probe.sh"), "env"], env=e, capture_output=True, text=True, check=True)
    env = dict(line.split("=", 1) for line in p.stdout.splitlines())
    probe = str(scratch / "lean" / "fournode-probe")
    for k in ("NEMO_RL_DIR", "MEGATRON_DIR", "BRIDGE_DIR", "VLLM_RUNNER", "VLLM_APERTUS2", "EDF_DIR", "OUT",
              "TRITON_CACHE"):
        assert env[k].startswith(probe + "/"), k
    assert env["EDF_NAME"] != "nemo-rl-apertus2" and env["SHARED_PROJECTS"] == str(tmp_path / "home" / "projects")
    assert env["RECIPE"].endswith("recipes/apertus2-chonk-sftv02-18315-vllm-adamw-code-4n-colocated.yaml")
    assert Path(env["RECIPE"]).is_file()
    assert env["EXTRA_OVERRIDES"] == ("grpo.max_num_steps=5 checkpointing.save_period=3 "
                                      "policy.generation.vllm_cfg.gpu_memory_utilization=0.7 x=1")
    assert env["DRIVER_RETRIES"] == "0" and env["PARTITION"] == "normal" and env["SKIP_SETUP"] == "1"
    # TMPDIR only inside setup(): sbatch would hand it to the job (Ray's session folder and sockets)
    text = (HERE / "probe.sh").read_text()
    setup_body = text.split("\nsetup() {\n", 1)[1].split("\n}\n", 1)[0]
    assert "TMPDIR=" in setup_body and "TMPDIR=" not in text.replace(setup_body, "")
    # the overlays run.sh sets for this task (the EDF names them)
    assert env["PYDEPS"] == f"{scratch}/nemo-rl-apertus2/pydeps-gym-apertus2" and env["PYDEPS_GYM"] == "1"
    assert env["PYDEPS_APERTUS2"].startswith(f"{scratch}/nemo-rl-apertus2/apertus-common-")
    assert len(env["PYDEPS_APERTUS2"].rsplit("-", 1)[1]) == 9
    assert env["DRIVER_PYTHON"] == f"bash {HERE}/probe_driver.sh" and env["NUMA_WATCH"] == f"{HERE}/probe_watch.sh"
    assert (HERE / "../../../apertus2/driver.sh").resolve().is_file()
    assert (HERE / "../../../numa_watch.sh").resolve().is_file()


def test_probe_refuses_a_folder_outside_scratch(tmp_path):
    e = dict(os.environ, SCRATCH=str(tmp_path / "scratch"), PROBE=str(tmp_path / "elsewhere"))
    p = subprocess.run(["bash", str(HERE / "probe.sh"), "env"], env=e, capture_output=True, text=True)
    assert p.returncode == 1 and "not under SCRATCH" in p.stderr


def test_edf_dir_names_the_file_by_path(tmp_path):
    lib = HERE / "../../../container.sh"
    script = f'source {lib}; EDF_NAME=n; container_edf_ref; EDF_DIR={tmp_path}/e; container_edf_ref'
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    assert p.stdout.split() == ["n", f"{tmp_path}/e/n.toml"]


# ---- report.py on a made-up job

def varint(n):
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def field(num, wt, payload):
    key = varint(num << 3 | wt)
    if wt == 0:
        return key + varint(payload)
    if wt == 2:
        return key + varint(len(payload)) + payload
    return key + payload


def event(step, tag, value):
    val = field(1, 2, tag.encode()) + field(2, 5, struct.pack("<f", value))
    rec = field(1, 1, struct.pack("<d", time.time())) + field(2, 0, step) + field(5, 2, field(1, 2, val))
    return struct.pack("<Q", len(rec)) + b"\0" * 4 + rec + b"\0" * 4


def fake_job(tmp, *, gpu_peak_mib=88000, restored=RANKS, oom=False, host_low=40.0):
    out = tmp / "out"
    run, job = "fournode-probe-x", "123"
    jl = out / "slurm" / f"{job}-logs"
    jl.mkdir(parents=True)
    (out / "slurm" / f"{job}-{run}.out").write_text("")
    (out / "slurm" / f"{job}-{run}.preflight.txt").write_text("")
    t0 = int(time.time()) - 3600
    ld = out / "logs" / run
    (ld / "exp_001" / "tensorboard").mkdir(parents=True)
    (ld / "probe-phases.tsv").write_text(f"{t0}\tA\tstart steps=3\n{t0 + 1500}\tA\tend rc=0\n"
                                         f"{t0 + 1600}\tB\tstart steps=5\n{t0 + 3000}\tB\tend rc=0\n")
    ev = b"".join(event(s, "timing/train/total_step_time", 900.0 + s) + event(s, "timing/train/generation", 400.0)
                  + event(s, "timing/train/checkpointing", 60.0) for s in range(1, 6))
    (ld / "exp_001" / "tensorboard" / "events.out.tfevents.1.nid1.1.0").write_bytes(ev)
    driver = ["ScratchAdamW moments: reset (checkpoint None)"] * RANKS + \
             [f"[probe_driver] 2026-11-25T10:00:00 phase B start steps=5"] + \
             [f"ScratchAdamW moments: restored (checkpoint /x/step_3)"] * restored
    if oom:
        driver.append("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
    (jl / "ray-driver.log").write_text("\n".join(driver) + "\n")
    ts = datetime_str(t0 + 2000)
    for n in range(4):
        node = f"nid00{n}"
        (jl / f"gpumem-{node}.csv").write_text("".join(
            f"{ts}, {g}, {gpu_peak_mib if (n, g) == (2, 1) else 70000}, 97871, 90\n" for g in range(4)))
        (jl / f"hostmem-{node}.tsv").write_text(
            f"{t0}\tMemTotal=460.0\tMemAvailable=200.0\tMemFree=190.0\tShmem=10.0\tcpufree0=50.0\tcpufree1=50.0\n"
            f"{t0 + 5}\tMemTotal=460.0\tMemAvailable={host_low if n == 3 else 100.0}\tMemFree=30.0\tShmem=10.0"
            f"\tcpufree0=8.0\tcpufree1=9.0\n")
        (jl / f"numa-hbm-{node}.log").write_text(
            "10:00:00 cpufree0=10G | node4 used=1.0G file=0.0G anon=0.1G shmem=0.0G cpufree1=9G\n")
    ck = out / "results" / run
    for step in (3, 5):
        for r in range(RANKS):
            d = ck / f"step_{step}" / "policy" / "weights" / "optimizer-scratch" / f"rank{r:05d}" / "hdo0"
            d.mkdir(parents=True)
            (d / "manifest.json").write_text("{}")
            (d / "g0.bin").write_bytes(b"\0" * 1000)
        (ck / f"step_{step}" / "policy" / "weights" / "model.bin").write_bytes(b"\0" * 2000)
    for r in range(RANKS):
        d = ck / "scratch-adam-live" / f"rank{r:05d}"
        d.mkdir(parents=True)
        (d / "hdo0.stats.jsonl").write_text("".join(json.dumps(
            {"step": s, "seconds": 80.0 + r, "compute_seconds": 30.0, "io_GBps": 1.5, "params": 10,
             "host": f"nid00{r // 4}", "generation": s - 1}) + "\n" for s in range(1, 6)))
    return out, job


def datetime_str(epoch):
    from datetime import datetime
    return datetime.fromtimestamp(epoch).strftime("%Y/%m/%d %H:%M:%S.000")


def run_report(out, job, capsys):
    assert report.main([str(out), job]) == 0
    return capsys.readouterr().out


def test_report_go(tmp_path, capsys):
    out, job = fake_job(tmp_path)
    text = run_report(out, job, capsys)
    assert text.startswith("Probe verdict: GO"), text
    assert "B" in [line for line in text.splitlines() if "nid002" in line and " 1 " in line][0]   # phase at peak
    assert "16 reset" in text and "16 restored" in text
    for part in ("GPU", "host", "resume", "checkpoint", "scratch"):
        assert any(line.split()[:2] == [part, "GO"] for line in text.splitlines()), part
    assert "total_step_time" in text.split("Phase times")[1].splitlines()[2]


def test_report_flags_problems(tmp_path, capsys):
    out, job = fake_job(tmp_path, gpu_peak_mib=97000, restored=10, host_low=6.0)
    text = run_report(out, job, capsys)
    assert text.startswith("Probe verdict: CHECK"), text
    heads = {tuple(line.split()[:2]) for line in text.splitlines()[:7]}
    assert {("GPU", "CHECK"), ("host", "CHECK"), ("resume", "CHECK")} <= heads


def test_report_oom_is_no_go(tmp_path, capsys):
    out, job = fake_job(tmp_path, oom=True)
    assert run_report(out, job, capsys).startswith("Probe verdict: NO-GO")
