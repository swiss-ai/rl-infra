"""A stand-in for a Gym model server's Anthropic route, for probing what Claude Code sends (login node, no GPU).

Serves `POST /v1/messages` (also under a rollout's capture prefix `/ng-rollout/<id>/training-token-capture`) the way
`nemo_gym/base_responses_api_model.py` `messages()` does: each reply is built as a Responses result (output items) and
turned into Anthropic JSON and SSE by Gym's own `AnthropicConverter`, so ids, block shapes and event order are the real
server's. The replies follow a script (one per scenario, below). Every request, any method or path, is logged with its
headers and body; any path other than `/v1/messages` gets 404, as from Gym's FastAPI app.

Needs the Gym checkout on sys.path (probe.py puts it there) and its dependencies.
"""

import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.openai_utils import NeMoGymResponse


_CAPTURE_PREFIX = re.compile(r"^/ng-rollout/[^/]+/training-token-capture(?=/)")
_CONVERTER = AnthropicConverter()
_SECRET_HEADERS = {"authorization", "x-api-key"}

# Messages are "main loop" when they offer tools; Claude Code's other calls (titles, summaries, ...) do not.
LONG_OUTPUT_CHARS = 30000


def _text(text: str) -> dict:
    return {
        "type": "message",
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _call(name: str, arguments: Any) -> dict:
    return {
        "type": "function_call",
        "id": f"fc_{uuid.uuid4().hex[:12]}",
        "call_id": f"call_{uuid.uuid4().hex[:12]}",
        "name": name,
        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
        "status": "completed",
    }


def _reasoning(text: str) -> dict:
    return {"type": "reasoning", "id": f"rs_{uuid.uuid4().hex[:12]}", "summary": [{"type": "summary_text", "text": text}]}


class Reply:
    """One scripted answer: Responses output items and whether the reply was cut by the output limit, or an
    HTTP error instead."""

    def __init__(self, items: list[dict] | None = None, cut: bool = False, http_error: Optional[tuple[int, dict]] = None):
        self.items = items or []
        self.cut = cut
        self.http_error = http_error


def _basic(k: int, workdir: str, thinking: bool = False) -> Reply:
    steps = [
        [_text("Let me look at the files."), _call("Bash", {"command": "ls", "description": "List files"})],
        [_call("Read", {"file_path": f"{workdir}/hello.py"})],
        [_call("Edit", {"file_path": f"{workdir}/hello.py", "old_string": "Hello", "new_string": "Hi"})],
        [_text("Done: hello.py now prints Hi.")],
    ]
    items = steps[min(k, len(steps) - 1)]
    if thinking:
        items = [_reasoning(f"Step {k}: what next?")] + items
    return Reply(items)


def _overloaded(k: int, workdir: str, state: dict) -> Reply:
    """The first main-loop call gets 529 overloaded, later ones the basic chain."""
    if not state.get("overloaded_sent"):
        state["overloaded_sent"] = True
        return Reply(http_error=(529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}))
    return _basic(k, workdir)


SCENARIOS: dict[str, Callable[[int, str, dict], Reply]] = {
    # ls, Read, Edit, final text: the normal chain
    "basic": lambda k, wd, st: _basic(k, wd),
    # the same with a thinking block before each reply (Apertus 2 reasons on every turn): echoed or dropped?
    "thinking": lambda k, wd, st: _basic(k, wd, thinking=True),
    # a tool call whose input is what Gym would send for arguments that are not JSON (proposed patch 0049)
    "badargs": lambda k, wd, st: Reply(
        [_call("Bash", {"_ng_unparsed_arguments": "ls -la"})] if k == 0 else [_text("Done.")]
    ),
    # the model server's context-overflow placeholder: no content, stop_reason max_tokens
    "maxtokens": lambda k, wd, st: Reply([], cut=True) if k == 0 else Reply([_text("Done.")]),
    # a long tool result followed by two more calls: is the earlier result ever rewritten?
    "longoutput": lambda k, wd, st: Reply(
        [
            [_call("Bash", {"command": f"python3 -c \"print('x' * {LONG_OUTPUT_CHARS})\"", "description": "Print"})],
            [_call("Bash", {"command": "echo second", "description": "Echo"})],
            [_call("Bash", {"command": "echo third", "description": "Echo"})],
            [_text("Done.")],
        ][min(k, 3)]
    ),
    # every offered tool once, with only its required fields: which echoes differ from what the model sent?
    "tools": lambda k, wd, st: Reply(
        [
            [_call("Glob", {"pattern": "*.py"})],
            [_call("Grep", {"pattern": "Hello"})],
            [_call("Read", {"file_path": f"{wd}/hello.py"})],
            [_call("Bash", {"command": "cat hello.py"})],
            [_call("Write", {"file_path": f"{wd}/notes.txt", "content": "x\n"})],
            [_call("Edit", {"file_path": f"{wd}/hello.py", "old_string": "Hello", "new_string": "Hi"})],
            [_text("Done.")],
        ][min(k, 6)]
    ),
    # inputs a policy may write that the tool schema does not take as given: an unknown key, a number as a string,
    # an explicit default, a key order other than the schema's
    "quirks": lambda k, wd, st: Reply(
        [
            [_call("Read", {"file_path": f"{wd}/hello.py", "foo": 1})],
            [_call("Read", {"file_path": f"{wd}/hello.py", "limit": "2"})],
            [_call("Bash", {"command": "echo hi", "timeout": "5000"})],
            [_call("Edit", {"replace_all": False, "new_string": "Hi", "old_string": "Hello", "file_path": f"{wd}/hello.py"})],
            [_text("Done.")],
        ][min(k, 4)]
    ),
    # what Claude Code 2.1.326 rewrote in its echoes on the cluster (rl-infra jobs 1061238, 1061276): a command that
    # starts with `cd <its cwd> && `, a written file with trailing whitespace, an Edit that leaves out replace_all
    "rewrites": lambda k, wd, st: Reply(
        [
            [_call("Bash", {"command": f"cd {wd} && ls", "description": "List"})],
            [_call("Write", {"file_path": f"{wd}/fib.py", "content": "def f(n):  \n    a, b = 0, 1\n    \n    return a\n"})],
            [_call("Edit", {"file_path": f"{wd}/hello.py", "old_string": "Hello", "new_string": "Hi"})],
            [_text("Done.")],
        ][min(k, 3)]
    ),
    # the second main-loop call is answered 400 "prompt is too long", as Anthropic's API answers a context overflow:
    # does Claude Code end cleanly (one call, no retry, no compaction)?
    "toolong": lambda k, wd, st: (
        _basic(k, wd)
        if k == 0
        else Reply(
            http_error=(
                400,
                {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "prompt is too long: 70000 tokens > 65536 maximum"},
                },
            )
        )
    ),
    # the same with the body Gym's vllm_model passes through under propagate_context_overflow_errors: vLLM's own
    # 400 (OpenAI error shape), not an Anthropic error
    "toolong_vllm": lambda k, wd, st: (
        _basic(k, wd)
        if k == 0
        else Reply(
            http_error=(
                400,
                {
                    "object": "error",
                    "message": "This model's maximum context length is 65536 tokens. However, you requested 70000 "
                    "tokens in the messages, Please reduce the length of the messages. None",
                    "type": "BadRequestError",
                    "param": None,
                    "code": 400,
                },
            )
        )
    ),
    # the first main-loop call is answered 529 overloaded: does Claude Code retry?
    "overloaded": _overloaded,
    # the basic chain under a turn budget (probe.py passes max_turns 2)
    "maxturns": lambda k, wd, st: _basic(k, wd),
}


class Stub:
    def __init__(self, scenario: str, workdir: str, log_path: str):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario}")
        self.scenario = scenario
        self.workdir = workdir
        self.log_path = log_path
        self.state: dict = {}
        self.lock = threading.Lock()
        self.served: list[dict] = []  # per /v1/messages answer: its request index, id and Responses output items
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "Stub":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _log(self, record: dict) -> int:
        with self.lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            self.state["n"] = self.state.get("n", 0) + 1
            return self.state["n"] - 1

    def reply_for(self, body: dict) -> tuple[Reply, bool]:
        main = bool(body.get("tools"))
        if not main:
            return Reply([_text("OK")]), False
        k = sum(1 for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "assistant")
        with self.lock:
            return SCENARIOS[self.scenario](k, self.workdir, self.state), True

    def _handler(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # quiet; everything is in the log file
                pass

            def _record(self, body_raw: bytes) -> tuple[int, Any, str]:
                path, _, query = self.path.partition("?")
                rest = _CAPTURE_PREFIX.sub("", path)
                try:
                    body = json.loads(body_raw) if body_raw else None
                except ValueError:
                    body = body_raw.decode(errors="replace")
                headers = {
                    k.lower(): ("<set>" if k.lower() in _SECRET_HEADERS else v) for k, v in self.headers.items()
                }
                index = stub._log(
                    {
                        "t": time.time(),
                        "method": self.command,
                        "path": path,
                        "route": rest,
                        "prefixed": rest != path,
                        "query": query,
                        "headers": headers,
                        "body": body,
                    }
                )
                return index, body, rest

            def _send(self, code: int, payload: bytes, ctype: str = "application/json") -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(payload)

            def _not_found(self) -> None:
                self._send(404, b'{"detail":"Not Found"}')

            def do_GET(self):  # noqa: N802
                self._record(b"")
                self._not_found()

            def do_HEAD(self):  # noqa: N802
                self._record(b"")
                self._not_found()

            def do_CONNECT(self):  # noqa: N802
                # probe.py makes the stub the CLI's HTTPS proxy: any connection to the outside is logged and refused
                self._record(b"")
                self._send(403, b"")

            def do_POST(self):  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                index, body, route = self._record(raw)
                if route != "/v1/messages" or not isinstance(body, dict):
                    self._not_found()
                    return
                reply, main = stub.reply_for(body)
                if reply.http_error is not None:
                    code, err = reply.http_error
                    self._send(code, json.dumps(err).encode())
                    return
                response = NeMoGymResponse.model_validate(
                    {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "created_at": time.time(),
                        "model": "policy",
                        "object": "response",
                        "output": reply.items,
                        "parallel_tool_calls": True,
                        "tool_choice": "auto",
                        "tools": [],
                        "status": "incomplete" if reply.cut else "completed",
                        "incomplete_details": {"reason": "max_output_tokens"} if reply.cut else None,
                        "usage": {
                            "input_tokens": len(raw) // 4,
                            "output_tokens": 0 if reply.cut and not reply.items else 20,
                            "total_tokens": len(raw) // 4 + 20,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens_details": {"reasoning_tokens": 0},
                        },
                    }
                )
                anthropic = _CONVERTER.responses_to_anthropic_response(response, model=body.get("model") or "policy")
                with stub.lock:
                    stub.served.append(
                        {"request": index, "id": anthropic["id"], "main": main, "output_items": reply.items,
                         "stop_reason": anthropic.get("stop_reason")}
                    )
                if body.get("stream"):
                    payload = "".join(_CONVERTER.anthropic_response_to_sse(anthropic)).encode()
                    self._send(200, payload, "text/event-stream")
                else:
                    self._send(200, json.dumps(anthropic).encode())

        return Handler
