"""Local tests of the stage-1 tools at tiny sizes (no GPU, no vLLM, no Slurm): pytest stage1_test.py"""
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conc_client  # noqa: E402
import host_bench  # noqa: E402
import report  # noqa: E402


def test_adam_and_merge(tmp_path):
    host_bench.main(["--out", str(tmp_path), "--threads", "2", "adam", "--params-m", "0.5", "--tensor-m", "0.2",
                     "--reps", "2", "--variants", "foreach,fused"])
    r = json.loads((tmp_path / "adam.0.json").read_text())
    assert r["foreach_Gparams_per_s"] > 0
    assert "fused_Gparams_per_s" in r or "fused_error" in r
    host_bench.main(["--out", str(tmp_path), "facts"])
    host_bench.main(["--out", str(tmp_path), "merge"])
    merged = json.loads((tmp_path / "host.json").read_text())
    assert {"adam", "facts"} <= set(merged)


def test_io(tmp_path):
    host_bench.main(["--out", str(tmp_path), "io", "--dir", str(tmp_path / "d"), "--tag", "t", "--streams", "2,4",
                     "--total-gb", "0.004", "--block-mb", "0.25", "--mixed", "--no-direct"])
    r = json.loads((tmp_path / "io-t.json").read_text())
    assert [x["streams"] for x in r["runs"]] == [2, 4]
    for x in r["runs"]:
        assert x["write"]["GBps"] > 0 and x["read"]["GBps"] > 0 and not x["write"]["errors"] and not x["read"]["errors"]
        assert x["read"]["bytes"] == x["write"]["bytes"] == x["file_bytes"] * x["streams"]
        assert x["mixed"]["read"] > 0 and x["mixed"]["write"] > 0
    assert not list((tmp_path / "d").glob("*.bin"))  # test files removed


@pytest.mark.parametrize("fused", [False, True])
def test_stream_adam_updates_in_place(tmp_path, fused):
    args = ["--out", str(tmp_path), "--threads", "2", "--ranks", "1", "stream-adam", "--dir", str(tmp_path / "s"),
            "--params-m", "0.3", "--chunk-m", "0.05", "--buffers", "3", "--readers", "2", "--writers", "2",
            "--steps", "2", "--block-mb", "0.25", "--no-direct"] + (["--fused"] if fused else [])
    try:
        host_bench.main(args)
    except RuntimeError as error:  # a torch without fused CPU AdamW
        pytest.skip(str(error))
    r = json.loads((tmp_path / "stream-adam.0.json").read_text())
    assert r["verify"]["ok"], r["verify"]
    assert len(r["steps"]) == 2 and all(not s["errors"] for s in r["steps"])
    assert r["chunks"] == 0.3e6 // (0.05e6 // 1024 * 1024)
    assert not list((tmp_path / "s").glob("state-*.bin"))  # chunk files removed


def test_stream_adam_state_persists_between_steps(tmp_path):
    """Two streamed steps leave on disk what two in-memory AdamW steps give (step 2 reads what step 1 wrote)."""
    import torch

    chunk = 2048
    torch.manual_seed(0)
    host_bench.main(["--out", str(tmp_path), "--threads", "1", "--ranks", "1", "stream-adam", "--dir", str(tmp_path / "s"),
                     "--params-m", str(chunk / 1e6), "--chunk-m", str(chunk / 1e6), "--buffers", "1", "--readers", "1",
                     "--writers", "1", "--steps", "2", "--block-mb", "0.0625", "--no-direct", "--keep"])
    flat = torch.frombuffer(bytearray((tmp_path / "s" / "state-r0-0000.bin").read_bytes()), dtype=torch.float32)
    torch.manual_seed(0)                      # the same draws as host_bench: the gradient, then the initial master
    grad = torch.randn(chunk) * 1e-3
    p = torch.nn.Parameter(torch.randn(chunk))
    opt = torch.optim.AdamW([p], lr=5e-6, betas=(0.9, 0.999), eps=1e-14, weight_decay=0.01, foreach=False)
    for _ in range(2):
        p.grad = grad
        opt.step()
    st = opt.state[p]
    assert torch.allclose(flat[:chunk], p.detach(), rtol=0, atol=1e-7)
    assert torch.allclose(flat[chunk:2 * chunk], st["exp_avg"], rtol=1e-6, atol=0)
    assert torch.allclose(flat[2 * chunk:], st["exp_avg_sq"], rtol=1e-6, atol=0)
    assert not torch.equal(flat[:chunk], torch.zeros(chunk))


# ---------------------------------------------------------------------------------------------------------------
# conc_client against a stand-in vLLM server


class FakeVLLM(BaseHTTPRequestHandler):
    gen = 0.0
    prompt = 0.0
    running = 0
    lock = threading.Lock()
    calls = []

    def log_message(self, *a):
        pass

    def reply(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        FakeVLLM.calls.append(("GET", self.path))
        if self.path == "/metrics":
            c = FakeVLLM
            text = "\n".join([
                "# HELP vllm:generation_tokens_total x",
                f'vllm:generation_tokens_total{{engine="0"}} {c.gen / 2}',
                f'vllm:generation_tokens_total{{engine="1"}} {c.gen / 2}',
                f'vllm:prompt_tokens_total{{engine="0"}} {c.prompt}',
                f'vllm:num_requests_running{{engine="0"}} {c.running}',
                'vllm:num_requests_waiting{engine="0"} 0',
                'vllm:kv_cache_usage_perc{engine="0"} 0.25',
                'vllm:kv_cache_usage_perc{engine="1"} 0.75',
                'vllm:num_preemptions_total{engine="0"} 0',
            ])
            self.reply(200, text, "text/plain")
        else:
            self.reply(200, {"is_sleeping": False})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        FakeVLLM.calls.append(("POST", self.path))
        if self.path == "/v1/completions":
            with FakeVLLM.lock:
                FakeVLLM.running += 1
            time.sleep(0.05)
            k = min(body["max_tokens"], 40)
            with FakeVLLM.lock:
                FakeVLLM.running -= 1
                FakeVLLM.gen += k
                FakeVLLM.prompt += len(body["prompt"])
            self.reply(200, {"choices": [{"token_ids": list(range(k)), "finish_reason": "length" if k == body["max_tokens"] else "stop"}]})
        else:
            self.reply(200, {"ok": True})


@pytest.fixture
def fake_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    FakeVLLM.calls.clear()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_metrics_parse():
    m = conc_client.parse_metrics('vllm:generation_tokens_total{a="1"} 10\nvllm:generation_tokens_total{a="2"} 5\n'
                                  'vllm:gpu_cache_usage_perc{a="1"} 0.2\nvllm:gpu_cache_usage_perc{a="2"} 0.4\nother 3\n')
    assert m["vllm:generation_tokens_total"] == 15 and abs(m["kv_usage"] - 0.3) < 1e-9


def test_sweep_and_sleepwake(tmp_path, fake_server):
    conc_client.main(["--url", fake_server, "--out", str(tmp_path), "sweep", "--levels", "2,4", "--gpus", "2",
                      "--warm", "0.5", "--measure", "1.5", "--every", "0.2", "--fake-prompts", "--max-new-tokens", "30"])
    levels = json.loads((tmp_path / "levels.json").read_text())
    assert [lv["level_per_gpu"] for lv in levels] == [2, 4]
    assert all(lv["gen_tok_per_s_per_gpu"] > 0 and lv["finished"] > 0 for lv in levels)
    assert levels[1]["gen_tok_per_s"] > levels[0]["gen_tok_per_s"]  # twice the requests in flight, same latency
    assert all(lv["cut_share"] == 1.0 for lv in levels)              # max_tokens 30 < 40: every request cut
    assert abs(levels[0]["mean_kv_usage"] - 0.5) < 1e-9
    conc_client.main(["--url", fake_server, "--out", str(tmp_path), "sleepwake", "--drain", "5"])
    sw = json.loads((tmp_path / "sleepwake.json").read_text())
    assert [s["step"] for s in sw["steps"]] == ["generate", "sleep1", "is_sleeping", "wake1", "generate", "sleep2",
                                               "wake2", "reload_weights", "generate"]
    assert all(s["status"] == 200 for s in sw["steps"])
    assert ("POST", "/sleep?level=2") in FakeVLLM.calls


def test_sweep_deadline_stops(tmp_path, fake_server):
    conc_client.main(["--url", fake_server, "--out", str(tmp_path), "sweep", "--levels", "1,2", "--gpus", "1",
                      "--warm", "0.3", "--measure", "0.5", "--every", "0.2", "--fake-prompts",
                      "--deadline", str(time.time() + 1.0)])
    assert len(json.loads((tmp_path / "levels.json").read_text())) == 1


# ---------------------------------------------------------------------------------------------------------------
# the whole host job with stand-in srun and lfs, and the report


def test_host_job_and_report(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # srun: drops its own options; with --ntasks=4 runs the command 4 times in parallel with SLURM_LOCALID set
    (bin_dir / "srun").write_text("""#!/bin/bash
n=1
while [[ $# -gt 0 && $1 == --* ]]; do
  [[ $1 == --ntasks=* ]] && n=${1#--ntasks=}
  [[ $1 == --environment=* && $1 != --environment=nemo-rl-apertus2 ]] && { echo "srun: wrong container $1" >&2; exit 9; }
  shift
done
pids=()
for ((i = 0; i < n; i++)); do SLURM_LOCALID=$i "$@" & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; exit $rc
""")
    (bin_dir / "lfs").write_text("#!/bin/bash\necho lfs \"$@\" >> \"$LFS_LOG\"\n")
    (bin_dir / "python3").write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    scratch = tmp_path / "scratch"
    code = scratch / "lean/fournode-stage1/code"
    code.mkdir(parents=True)
    for f in ("host_bench.py",):
        (code / f).write_text((HERE / f).read_text())
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", SCRATCH=str(scratch), SLURM_JOB_ID="77",
               ENVIRONMENT="BATCH",   # Slurm sets this in every batch job
               SKIP="c2c gradsum", IO_GB="0.03", IO_BLOCK_MB="0.25", ADAM_PARAMS_M="0.2", STREAM_PARAMS_M="0.2", STREAM_CHUNK_M="0.05",
               START_DELAY="1", LFS_LOG=str(tmp_path / "lfs.log"))
    out = subprocess.run(["bash", str(HERE / "host_job.sbatch")], env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    o = scratch / "lean/fournode-stage1/out/77-host"
    merged = json.loads((o / "host.json").read_text())
    assert {"facts", "adam", "io-default", "io-c4-16m", "stream-adam"} <= set(merged), sorted(merged)
    assert len(merged["adam"]) == 4 and len(merged["stream-adam"]) == 4
    assert all(r["write"]["bytes"] > 0 and r["read"]["bytes"] == r["write"]["bytes"] for r in merged["io-default"]["runs"])
    assert all(r["verify"]["ok"] for r in merged["stream-adam"])
    assert "setstripe -c 4 -S 16M" in (tmp_path / "lfs.log").read_text()
    assert not (o / "data").exists()
    assert {p.name for p in o.glob("done.*")} == {"done.facts", "done.adam", "done.io-default", "done.io-c4", "done.stream-adam"}
    # a requeued job skips what is done
    again = subprocess.run(["bash", str(HERE / "host_job.sbatch")], env=env, capture_output=True, text=True, timeout=600)
    assert again.returncode == 0 and again.stdout.count("skipped (done or SKIP)") == 7, again.stdout

    # report: the host results plus a stand-in vLLM run
    v = scratch / "lean/fournode-stage1/out/78-vllm-kvfp8/r0"
    v.mkdir(parents=True)
    (v / "levels.json").write_text(json.dumps([{"level_per_gpu": 64, "gen_tok_per_s_per_gpu": 2000.0, "mean_running": 250,
                                                "mean_kv_usage": 0.4, "peak_kv_usage": 0.6, "preemptions": 0,
                                                "cut_share": 0.1}]))
    (v / "server.txt").write_text("weights=fp8 sleep_mode=1 ready_s=300\n")
    (v / "sleepwake.json").write_text(json.dumps({"steps": [{"step": "sleep2", "status": 200, "seconds": 2.0, "gpu_mem_mib": [1, 1, 1, 1]}]}))
    md = tmp_path / "r.md"
    report.main([str(scratch / "lean/fournode-stage1/out"), "--md", str(md)])
    text = md.read_text()
    assert "Streamed Adam step from scratch" in text and "AdamW in host memory, foreach" in text
    assert "| fp8 | fp8 | 64 | 2,000 | 1.12x |" in text
    assert "sleep2 200 2.0 s" in text
