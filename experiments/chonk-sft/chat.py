#!/usr/bin/env python3
"""Talk to an Apertus 2 checkpoint that serve.sbatch is serving, through vLLM's native completions.

Run it with the venv from setup.sh (apertus-common needs Python 3.13), on the login node:
    venv/bin/python chat.py                              # interactive
    venv/bin/python chat.py --prompts prompts.jsonl      # one new conversation per line

Requests follow the format's engine rules (README.md): the prompt goes as token IDs so that no text
can become a control token, the special-ID mask goes as allowed_token_ids, and generation ends at
<|wait|>, the checkpoint's EOS. Replies are parsed from token IDs. A turn continues from the exact
tokens of the last one, never from re-encoded text.
"""
import argparse
import concurrent.futures
import json
import os
import resource
import sys
import time
import urllib.request
from pathlib import Path

# Login nodes default to 300 MB stacks and a 32 GB address space, which NumPy's one thread per core
# nearly fills on import, leaving room for 33 more. glibc reads the limit at start-up, hence the exec.
STACK_BYTES = 8 << 20
soft_stack, hard_stack = resource.getrlimit(resource.RLIMIT_STACK)
if soft_stack == resource.RLIM_INFINITY or soft_stack > STACK_BYTES:
    resource.setrlimit(resource.RLIMIT_STACK, (STACK_BYTES, hard_stack))
    os.execv(sys.executable, [sys.executable, *sys.argv])

from apertus_common import (  # noqa: E402
    Call,
    Conversation,
    Document,
    MessageCompleted,
    MessageStart,
    PayloadDelta,
    Result,
    StreamParseError,
    SystemPrompt,
    ToolSpec,
    User,
    load_encoding,
)

ROOT = Path(os.environ.get("ROOT", f"/scratch/{os.environ['USER']}/chonk-sft"))
MAX_BURSTS = 8  # for one --prompts line, so that a model that keeps calling a tool still ends


class Client:
    def __init__(self, args):
        self.args = args
        endpoint = args.endpoint or (ROOT / "endpoint").read_text().strip()
        self.endpoint = endpoint.rstrip("/")
        self.encoding = load_encoding(args.tokenizer)
        forbidden = set(self.encoding.generation_constraints.forbidden_token_ids)
        self.allowed = [i for i in range(self.encoding.vocab_size) if i not in forbidden]
        with urllib.request.urlopen(f"{self.endpoint}/v1/models", timeout=30) as response:
            self.model = json.load(response)["data"][0]["root"]

    def generate(self, prompt, on_event=None):
        """One burst. Returns (generated IDs, parsed output or None, parse error, finish reason)."""
        args = self.args
        body = {
            "model": "chonk",
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "allowed_token_ids": self.allowed,
            "return_token_ids": True,
            "skip_special_tokens": False,
            "stream": True,
        }
        if args.seed is not None:
            body["seed"] = args.seed
        request = urllib.request.Request(
            f"{self.endpoint}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        parser = self.encoding.output_parser()
        generated, error, finish = [], None, None

        def feed(tokens):
            nonlocal error
            if error is not None:
                return
            try:
                parser.push_tokens(tokens)
            except StreamParseError as failure:
                error = failure
            for event in parser.take_events():
                if on_event:
                    on_event(event)

        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for raw in response:
                line = raw.strip()
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                choice = json.loads(line[len(b"data: "):])["choices"][0]
                tokens = choice.get("token_ids") or []
                generated += tokens
                feed(tokens)
                finish = choice.get("finish_reason") or finish
        # vLLM may leave the EOS out of the IDs; the burst still ended at <|wait|>.
        wait = self.encoding.controls.wait
        if finish == "stop" and (not generated or generated[-1] != wait):
            generated.append(wait)
            feed([wait])
        output = None
        if error is None:
            try:
                output = parser.finish()
            except StreamParseError as failure:
                error = failure
        return generated, output, error, finish


def system_prompt(spec, args):
    tools = [ToolSpec.model_validate(tool) for tool in spec.get("tools", [])]
    return SystemPrompt.build(
        thinking=spec.get("thinking", args.thinking),
        identity=spec.get("identity", args.identity),
        behavior=spec.get("behavior", args.behavior),
        environment=spec.get("environment"),
        tools=tools,
    )


def run_prompts(client, args):
    prompts = [json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    tag = Path(client.model).name
    out = Path(args.out or ROOT / "results" / f"{tag}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    def one(spec):
        encoding = client.encoding
        history = encoding.encode_conversation(Conversation(system=system_prompt(spec, args)))
        turns = [[User(payload=text)] for text in spec.get("turns", [spec.get("user")])]
        turns[0][:0] = [Document.model_validate(document) for document in spec.get("documents", [])]
        bursts = []
        while turns and len(bursts) < MAX_BURSTS:
            items = turns.pop(0)
            prompt = history + encoding.encode_items(items)
            start = time.time()
            generated, output, error, finish = client.generate(prompt)
            bursts.append(
                {
                    "input": [item.to_json() for item in items],
                    "prompt_tokens": len(prompt),
                    "generated_tokens": len(generated),
                    "seconds": round(time.time() - start, 1),
                    "finish_reason": finish,
                    "output": output.to_json() if output is not None else None,
                    "parse_error": str(error) if error is not None else None,
                    "raw": encoding.decode_text(generated),
                }
            )
            if error is not None:
                break
            history = prompt + generated
            # Scripted tools: answer each call from the line's "results", by tool name.
            results = [
                Result(name=item.name, counter=item.counter, payload=spec["results"][item.name])
                for item in output.items
                if isinstance(item, Call) and item.name in spec.get("results", {})
            ]
            if results:
                turns.insert(0, results)
        return {"id": spec["id"], "model": client.model, "system": str(system_prompt(spec, args)), "bursts": bursts}

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool, out.open("w") as sink:
        futures = [pool.submit(one, spec) for spec in prompts]
        for future in concurrent.futures.as_completed(futures):  # saved as they end, so a stop keeps them
            result = future.result()
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            shapes = [
                ">".join(item["type"] for item in json.loads(burst["output"])["items"])
                if burst["output"] is not None
                else f"PARSE ERROR ({burst['finish_reason']})"
                for burst in result["bursts"]
            ]
            tokens = sum(burst["generated_tokens"] for burst in result["bursts"])
            seconds = sum(burst["seconds"] for burst in result["bursts"])
            print(f"{result['id']:<22} {tokens:>6} tok {seconds:>7.1f}s  {' | '.join(shapes)}")
    print(f"saved {out}")


def interact(client, args):
    print(f"{client.model}\nthinking={args.thinking}. /reset starts over, /quit leaves.\n")
    system = system_prompt({}, args)
    history = client.encoding.encode_conversation(Conversation(system=system))

    def show(event):
        if isinstance(event, MessageStart):
            print(f"[{event.type}{json.dumps(event.headers) if event.headers else ''}] ", end="", flush=True)
        elif isinstance(event, PayloadDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, MessageCompleted):
            print("\n")

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text == "/quit":
            return
        if text == "/reset":
            history = client.encoding.encode_conversation(Conversation(system=system))
            continue
        if not text:
            continue
        prompt = history + client.encoding.encode_items([User(payload=text)])
        start = time.time()
        generated, _, error, finish = client.generate(prompt, show)
        seconds = time.time() - start
        rate = len(generated) / seconds if seconds else 0.0
        if error is not None:
            print()
        print(f"({len(generated)} tokens, {seconds:.1f}s, {rate:.1f} tok/s, finish={finish})\n")
        if error is None:
            history = prompt + generated
        else:
            print(f"parse error, this turn is dropped from the history: {error}")
            print(f"raw output:\n{client.encoding.decode_text(generated)}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", help=f"default: the URL in {ROOT}/endpoint")
    parser.add_argument("--tokenizer", default=str(ROOT / "tokenizer"))
    parser.add_argument(
        "--prompts",
        help="JSONL: id, user or turns (a list), and optional thinking, identity, behavior, environment, tools, "
        "results (tool name to scripted output), documents",
    )
    parser.add_argument("--out", help="results file for --prompts")
    parser.add_argument("--workers", type=int, default=4, help="concurrent requests for --prompts")
    parser.add_argument("--thinking", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--identity")
    parser.add_argument("--behavior")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=float, default=3600, help="seconds without a token before giving up")
    args = parser.parse_args()
    client = Client(args)
    if args.prompts:
        run_prompts(client, args)
    else:
        interact(client, args)


if __name__ == "__main__":
    sys.exit(main())
