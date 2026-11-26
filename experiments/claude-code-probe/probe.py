"""Run Claude Code against stub_messages.py with the Gym adapter's exact command and environment, and check its
requests the way Gym's token capture would (login node, no GPU).

    python probe.py <gym checkout> <archive dir> <out dir> [scenario ...]

<gym checkout>: a Gym tree with patch 0048 (ClaudeCodeNemoGym) applied; <archive dir>: fetch-claude-code.sh's CC_DIR
(the probe unpacks the linux-arm64 or linux-x64 binary for this machine). For each scenario (stub_messages.SCENARIOS)
and profile (default, bare) it builds the adapter with an api_base under a capture prefix, takes the command of
`create_run_agent_commands` with the pod paths (binary, log, CLAUDE_CONFIG_DIR) moved under <out dir>, runs it in a
fresh git repository, and checks:
- every request: method and path (anything but /v1/messages would be marked incomplete under capture);
- the /v1/messages calls as the lineage store resolves them (nemo_gym.token_id_capture: assistant_fingerprint over a
  call's request items plus its output items, conversation_digest over the request items): one root and no
  unresolved call means one chain;
- system prompt and tool list identical across the main-loop calls; stream; max_tokens; thinking; beta headers;
- Claude Code's exit code and its stream-json `result` event; the served ids in the session JSONL.
Writes <out dir>/summary.json and the first main-loop request of each profile (first-request-<profile>.json, for
check B; it holds Claude Code's own system prompt, so it stays out of git).
"""

import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


GYM, ARCHIVES, OUT = (Path(a).resolve() for a in sys.argv[1:4])
SCENARIO_ARGS = sys.argv[4:]
sys.path.insert(0, str(GYM))
sys.path.insert(0, str(Path(__file__).parent))

from nemo_gym.anthropic_converter import AnthropicConverter  # noqa: E402
from nemo_gym.base_responses_api_model import _request_messages  # noqa: E402
from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint, conversation_digest  # noqa: E402
from stub_messages import SCENARIOS, Stub  # noqa: E402


cc = importlib.import_module("responses_api_agents.harbor_agent.custom_agents.claude_code_nemo_gym")
PLATFORM = "linux-arm64" if platform.machine() in ("aarch64", "arm64") else "linux-x64"
INSTRUCTION = "In this repository, change the greeting in hello.py from Hello to Hi."
CONVERTER = AnthropicConverter()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def install_binary(bin_dir: Path) -> Path:
    """The release binary from the archive dir, checked against the adapter's pinned digest."""
    sha256 = cc.KNOWN_BINARY_SHA256[(cc.DEFAULT_VERSION, PLATFORM)]
    dest = bin_dir / cc.DEFAULT_VERSION / "claude"
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    gz = ARCHIVES / cc.DEFAULT_VERSION / f"claude-{PLATFORM}.gz"
    subprocess.run(f"gzip -dc {gz} > {dest}.part", shell=True, check=True)
    if hashlib.sha256(Path(f"{dest}.part").read_bytes()).hexdigest() != sha256:
        raise SystemExit(f"{gz}: the unpacked binary does not match the pinned sha256")
    os.chmod(f"{dest}.part", 0o755)
    os.replace(f"{dest}.part", dest)
    return dest


def make_repo(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "hello.py").write_text("print('Hello')\n")
    (path / "README.md").write_text("A greeting.\n")
    git = ["git", "-c", "user.name=probe", "-c", "user.email=probe@localhost"]
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run([*git, "-C", str(path), "add", "-A"], check=True)
    subprocess.run([*git, "-C", str(path), "commit", "-qm", "init"], check=True)


def lineage(requests: list[dict], served: list[dict]) -> dict:
    """Resolve every /v1/messages call as the capture's lineage store would (see the module docstring)."""
    out_by_request = {s["request"]: s for s in served}
    rows, decisions = [], []
    for i, rec in enumerate(requests):
        if rec["route"] != "/v1/messages" or rec["method"] != "POST" or not isinstance(rec["body"], dict):
            continue
        try:
            items = _request_messages(CONVERTER.anthropic_request_to_responses(rec["body"]))
        except Exception as exc:  # what Gym would answer with a 500
            decisions.append({"request": i, "status": "converter_error", "error": f"{type(exc).__name__}: {exc}"})
            continue
        fp = assistant_fingerprint(items)
        if not fp:
            status, parent = "root", None
        else:
            matches = [
                r["request"]
                for r in rows
                if r["fp"] == fp and len(items) >= r["clen"] and conversation_digest(items[: r["clen"]]) == r["cdig"]
            ]
            status, parent = ("resolved", matches[0]) if len(matches) == 1 else ("unresolved", None)
            if len(matches) > 1:
                status = "ambiguous"
        decisions.append({"request": i, "status": status, "parent": parent, "main": bool(rec["body"].get("tools"))})
        if i in out_by_request:
            output = out_by_request[i]["output_items"]
            rows.append(
                {
                    "request": i,
                    "fp": assistant_fingerprint(list(items) + list(output)),
                    "clen": len(items),
                    "cdig": conversation_digest(items),
                }
            )
    count = lambda s: sum(1 for d in decisions if d["status"] == s)  # noqa: E731
    return {
        "calls": len(decisions),
        "roots": count("root"),
        "resolved": count("resolved"),
        "unresolved": count("unresolved") + count("ambiguous"),
        "converter_errors": count("converter_error"),
        "decisions": decisions,
    }


def echo_diffs(requests: list[dict], served: list[dict]) -> list[dict]:
    """For every tool call the stub served, how its echo (the tool_use with the same id in later requests) differs."""
    echoes = {}
    for rec in requests:
        body = rec["body"] if isinstance(rec["body"], dict) else {}
        for m in body.get("messages") or []:
            if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
                continue
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    echoes.setdefault(block["id"], block.get("input"))
    diffs = []
    for s in served:
        for item in s["output_items"]:
            if item.get("type") != "function_call" or item["call_id"] not in echoes:
                continue
            sent, echoed = json.loads(item["arguments"]), echoes[item["call_id"]]
            if sent == echoed:
                continue
            echoed = echoed if isinstance(echoed, dict) else {"<not an object>": echoed}
            diffs.append(
                {
                    "tool": item["name"],
                    "added": {k: echoed[k] for k in echoed if k not in sent},
                    "removed": sorted(k for k in sent if k not in echoed),
                    "changed": {k: [sent[k], echoed[k]] for k in sent if k in echoed and sent[k] != echoed[k]},
                }
            )
    return diffs


def tool_results(body: dict) -> list[str]:
    out = []
    for m in body.get("messages") or []:
        for block in m.get("content") or [] if isinstance(m.get("content"), list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                content = block.get("content")
                out.append(content if isinstance(content, str) else json.dumps(content))
    return out


def analyze(run_dir: Path, stub: Stub, proc: subprocess.CompletedProcess) -> dict:
    requests = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines() if line.strip()]
    messages = [r for r in requests if r["route"] == "/v1/messages" and isinstance(r["body"], dict)]
    main = [r for r in messages if r["body"].get("tools")]
    side = [r for r in messages if not r["body"].get("tools")]
    first_user = None
    if main:
        content = (main[0]["body"].get("messages") or [{}])[0].get("content")
        if isinstance(content, list):
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            first_user = texts
        else:
            first_user = [content]
    # did any tool result change between consecutive main-loop calls?
    rewritten = False
    for a, b in zip(main, main[1:]):
        ra, rb = tool_results(a["body"]), tool_results(b["body"])
        if rb[: len(ra)] != ra:
            rewritten = True
    events = []
    for line in (run_dir / "claude-code.txt").read_text(errors="replace").splitlines():
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    served_ids = [s["id"] for s in stub.served]
    session_ids = []
    for jsonl in (run_dir / "sessions").rglob("*.jsonl"):
        for line in jsonl.read_text(errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg = rec.get("message") if isinstance(rec, dict) else None
            if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("id"):
                session_ids.append(msg["id"])
    return {
        "exit_code": proc.returncode,
        "requests_by_path": sorted({f"{r['method']} {r['route']}{'?' + r['query'] if r['query'] else ''}" for r in requests}),
        "unprefixed_requests": sum(1 for r in requests if not r["prefixed"]),
        "messages_calls": len(messages),
        "main_calls": len(main),
        "side_calls": [
            {"system": str(r["body"].get("system"))[:200], "max_tokens": r["body"].get("max_tokens")} for r in side
        ],
        "lineage": lineage(requests, stub.served),
        "tool_echo_diffs": echo_diffs(requests, stub.served),
        "system_variants": len({_digest(r["body"].get("system")) for r in main}),
        "tools_variants": len({_digest(r["body"].get("tools")) for r in main}),
        "tool_names": [t.get("name") for t in (main[0]["body"].get("tools") or [])] if main else [],
        "body_keys": sorted({k for r in messages for k in r["body"]}),
        "stream": sorted({bool(r["body"].get("stream")) for r in messages}),
        "max_tokens": sorted({r["body"].get("max_tokens") for r in messages}, key=str),
        "thinking_param": sorted({json.dumps(r["body"].get("thinking")) for r in messages}),
        "anthropic_beta": sorted({r["headers"].get("anthropic-beta", "") for r in messages}),
        "user_agent": sorted({r["headers"].get("user-agent", "") for r in requests}),
        "echoed_thinking_blocks": sum(
            1
            for r in main
            for m in r["body"].get("messages") or []
            if m.get("role") == "assistant" and isinstance(m.get("content"), list)
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")
        ),
        "cache_control_blocks": sum(json.dumps(r["body"]).count('"cache_control"') for r in messages),
        "tool_results_rewritten": rewritten,
        "first_user_text_blocks": [t[:300] for t in (first_user or [])],
        "instruction_is_first_user_text": bool(first_user) and INSTRUCTION in (first_user or []),
        "system_chars": len(json.dumps(main[0]["body"].get("system"))) if main else 0,
        "tools_chars": len(json.dumps(main[0]["body"].get("tools"))) if main else 0,
        "result_event": {k: result.get(k) for k in ("subtype", "is_error", "num_turns", "stop_reason")} if result else None,
        "served_ids": len(served_ids),
        "session_assistant_ids": len(session_ids),
        "session_ids_are_served_ids": bool(session_ids) and set(session_ids) <= set(served_ids),
        "last_session_id_is_last_served": bool(session_ids) and bool(served_ids) and session_ids[-1] == served_ids[-1],
        "stderr_tail": proc.stderr[-1500:] if proc.stderr else "",
    }


def run_one(binary: Path, scenario: str, profile: str) -> dict:
    run_dir = OUT / f"{profile}-{scenario}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    workdir = run_dir / "repo"
    make_repo(workdir)
    (run_dir / "home").mkdir()
    # the pod paths of the adapter, moved here
    cc.CLAUDE_BIN = str(binary)
    cc.LOG_FILE = str(run_dir / "claude-code.txt")
    cc.CONFIG_DIR = str(run_dir / "sessions")
    with Stub(scenario, str(workdir), str(run_dir / "requests.jsonl")) as stub:
        agent = cc.ClaudeCodeNemoGym(
            logs_dir=run_dir / "agent-logs",
            model_name="/capstor/scratch/models/policy-path",
            api_base=f"http://127.0.0.1:{stub.port}/ng-rollout/probe-{profile}-{scenario}/training-token-capture/v1",
            claude_archive_dir=str(ARCHIVES),
            claude_platform=PLATFORM,
            claude_timeout_sec=180,
            claude_profile=profile,
            claude_debug=True,
            max_turns=2 if scenario == "maxturns" else 20,
            claude_env=json.loads(os.environ.get("PROBE_CLAUDE_ENV") or "{}"),
            model_info={"max_input_tokens": 65536, "max_output_tokens": 16384},
            responses_create_params={"max_output_tokens": 16384},
        )
        (exec_input,) = agent.create_run_agent_commands(INSTRUCTION)
        (run_dir / "command.txt").write_text(exec_input.command)
        (run_dir / "env.json").write_text(json.dumps(exec_input.env, indent=1))
        # The stub is also the HTTPS/HTTP proxy, so any call that bypasses ANTHROPIC_BASE_URL (telemetry, feature
        # flags, update checks) shows up in the log as CONNECT <host> and is refused, instead of leaving this node.
        proxy = f"http://127.0.0.1:{stub.port}"
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(run_dir / "home"),
            "LANG": "C.UTF-8",
            "HTTPS_PROXY": proxy,
            "HTTP_PROXY": proxy,
            "https_proxy": proxy,
            "http_proxy": proxy,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            **exec_input.env,
        }
        try:
            proc = subprocess.run(
                ["bash", "-c", exec_input.command], cwd=workdir, env=env, capture_output=True, text=True, timeout=240
            )
        except subprocess.TimeoutExpired as exc:
            proc = subprocess.CompletedProcess(exc.cmd, -9, exc.stdout or "", f"probe timeout\n{exc.stderr or ''}")
        Path(run_dir / "stub-served.json").write_text(json.dumps(stub.served, indent=1))
        summary = analyze(run_dir, stub, proc)
    requests = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines() if line.strip()]
    first_main = next((r["body"] for r in requests if isinstance(r["body"], dict) and r["body"].get("tools")), None)
    if scenario == "basic" and first_main is not None:
        (OUT / f"first-request-{profile}.json").write_text(json.dumps(first_main, indent=1))
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    binary = install_binary(OUT / "bin")
    scenarios = SCENARIO_ARGS or list(SCENARIOS)
    results = {}
    for profile in cc.PROFILES:
        for scenario in scenarios:
            key = f"{profile}-{scenario}"
            print(f"== {key}", flush=True)
            results[key] = run_one(binary, scenario, profile)
            r = results[key]
            lin = r["lineage"]
            print(
                f"   exit {r['exit_code']}  paths {r['requests_by_path']}  calls {lin['calls']} (main {r['main_calls']},"
                f" side {len(r['side_calls'])})  roots {lin['roots']}  unresolved {lin['unresolved']}  converter errors"
                f" {lin['converter_errors']}  system/tools variants {r['system_variants']}/{r['tools_variants']}"
                f"  result {r['result_event']}",
                flush=True,
            )
    (OUT / "summary.json").write_text(json.dumps({"version": cc.DEFAULT_VERSION, "platform": PLATFORM, **results}, indent=1))
    print(f"== summary: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
