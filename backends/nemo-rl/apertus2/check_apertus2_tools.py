"""Encode OpenCode-shaped chat requests in the Apertus 2 format, as NeMo-RL's chat endpoint does.

The endpoint (vllm_worker_async.py, patch 0025) encodes a request without a token prefix with
encode_chat_prompt(model_dir, system_prompt=<system_prompt_file>, messages, tools). This does the
same for each request file, without a GPU, and prints the prompt length, what the tools cost, the
budget left in the context, and the rendered prompt. A second encoding adds one tool round trip
(assistant tool call + tool result) to the messages, to check that from_openai accepts it.

    python3.12 check_apertus2_tools.py MODEL_DIR SYSTEM_PROMPT_FILE CONTEXT REQUEST.json...

A request file is a chat-completions body: {"messages": [...], "tools": [...], ...}. Exits 1 when
an encoding fails or a prompt does not fit CONTEXT minus the request's max_tokens.
"""

import copy
import json
import sys
import traceback
from pathlib import Path

from tokenizers import Tokenizer

from nemo_rl.data.apertus2_format import encode_chat_prompt

ROUND_TRIP = [
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_check_0",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": "ls /app", "description": "List /app"})},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_check_0", "content": "fib.py\nREADME.md\n"},
]


def encode(model: str, system_prompt: str, messages: list, tools: list | None) -> list[int]:
    return encode_chat_prompt(model, system_prompt=system_prompt, messages=messages, tools=tools)


def main() -> int:
    model, system_prompt_file, context, *requests = sys.argv[1:]
    context = int(context)
    system_prompt = Path(system_prompt_file).read_text(encoding="utf-8")
    tokenizer = Tokenizer.from_file(str(Path(model) / "tokenizer.json"))
    print(f"== model {model}\n== system prompt {system_prompt.strip()}\n== context {context}")
    failed = False
    for path in requests:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        messages, tools = body["messages"], body.get("tools") or None
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 0
        names = [tool["function"]["name"] for tool in tools or []]
        print(f"\n==== {Path(path).name}: {len(messages)} messages "
              f"(roles {[m['role'] for m in messages]}), {len(names)} tools {names}, max_tokens {max_tokens}")
        try:
            ids = encode(model, system_prompt, messages, tools)
        except Exception:
            traceback.print_exc()
            print(f"FAIL {Path(path).name}: encode_chat_prompt raised")
            failed = True
            continue
        try:
            no_tools = len(encode(model, system_prompt, messages, None))
            tools_cost = f"{len(ids) - no_tools} (prompt without tools: {no_tools})"
        except Exception as error:
            tools_cost = f"n/a (encoding without tools raised {error!r})"
        # The endpoint clamps max_tokens to the context (_clamp_max_tokens): only no room at all fails.
        left = context - len(ids)
        note = " -- NO ROOM, FAIL" if left <= 0 else f" -- WARNING: max_tokens {max_tokens} clamped" if left < max_tokens else ""
        print(f"prompt tokens {len(ids)}; tools {tools_cost}; left in context {left}{note}")
        failed |= left <= 0
        print(f"first ids {ids[:8]} last ids {ids[-4:]}")
        print("---- rendered prompt")
        print(tokenizer.decode(ids, skip_special_tokens=False))
        print("---- end of rendered prompt")

        follow = copy.deepcopy(messages) + copy.deepcopy(ROUND_TRIP)
        if "bash" not in names:
            print("round trip: skipped (no bash tool)")
            continue
        try:
            follow_ids = encode(model, system_prompt, follow, tools)
        except Exception:
            traceback.print_exc()
            print(f"FAIL {Path(path).name}: round trip (assistant tool_calls + role tool) raised")
            failed = True
            continue
        print(f"round trip: prompt tokens {len(follow_ids)} (+{len(follow_ids) - len(ids)}); "
              f"starts with the first prompt: {follow_ids[: len(ids)] == ids}")
        print("---- rendered round trip (after the first prompt)")
        print(tokenizer.decode(follow_ids[len(ids) :], skip_special_tokens=False))
        print("---- end of rendered round trip")
    print("\n== " + ("FAIL" if failed else "OK: every request encodes and fits"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
