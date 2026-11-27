"""Runs INSIDE the sandbox (standard library only): plain HTTP answers and a WebSocket echo on one
port, so the node can test the OpenSandbox server proxy in both modes.

    python3 ws_server.py <port>
  GET /...                         -> 200 "hello from <pod> path=<path>"
  GET /... with Upgrade: websocket -> echo every text/binary message back; answer pings; close on close
"""

import asyncio
import base64
import hashlib
import socket
import struct
import sys
import time

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def log(msg: str) -> None:
    print(f"[ws_server {time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    b1, b2 = await reader.readexactly(2)
    opcode = b1 & 0x0F
    masked = b2 & 0x80
    length = b2 & 0x7F
    if length == 126:
        (length,) = struct.unpack("!H", await reader.readexactly(2))
    elif length == 127:
        (length,) = struct.unpack("!Q", await reader.readexactly(8))
    mask = await reader.readexactly(4) if masked else b"\0\0\0\0"
    data = bytearray(await reader.readexactly(length))
    if masked:
        for i in range(length):
            data[i] ^= mask[i % 4]
    return opcode, bytes(data)


def frame(opcode: int, data: bytes) -> bytes:
    head = bytes([0x80 | opcode])
    n = len(data)
    if n < 126:
        head += bytes([n])
    elif n < 65536:
        head += bytes([126]) + struct.pack("!H", n)
    else:
        head += bytes([127]) + struct.pack("!Q", n)
    return head + data


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    peer = writer.get_extra_info("peername")
    try:
        request = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
        writer.close()
        return
    lines = request.split("\r\n")
    method, path, _ = (lines[0].split(" ", 2) + ["", ""])[:3]
    headers = {k.strip().lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:] if l)}
    log(f"{peer} {method} {path} upgrade={headers.get('upgrade', '-')}")
    if headers.get("upgrade", "").lower() != "websocket":
        body = f"hello from {socket.gethostname()} path={path}\n".encode()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(body) + body)
        await writer.drain()
        writer.close()
        return
    accept = base64.b64encode(hashlib.sha1((headers["sec-websocket-key"] + GUID).encode()).digest()).decode()
    writer.write(
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n".encode()
    )
    await writer.drain()
    n = 0
    try:
        while True:
            opcode, data = await read_frame(reader)
            if opcode == 0x8:  # close
                writer.write(frame(0x8, data[:2]))
                await writer.drain()
                break
            if opcode == 0x9:  # ping
                writer.write(frame(0xA, data))
            elif opcode in (0x1, 0x2):
                n += 1
                writer.write(frame(opcode, data))
            await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionError) as e:
        log(f"{peer} websocket ended: {e!r}")
    log(f"{peer} websocket closed after {n} messages")
    writer.close()


async def main() -> None:
    port = int(sys.argv[1])
    server = await asyncio.start_server(handle, "0.0.0.0", port, limit=1 << 16)
    log(f"listening on 0.0.0.0:{port} ({socket.gethostname()})")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
