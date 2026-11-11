"""Runs on the NODE (standard library only): stand-in for policy_model, the far end of the tunnel.

    python3 model_stub.py <port>        (binds 127.0.0.1; the chisel client on the same node dials it)
  GET  .../stream?n=N&dt=S  -> chunked text/event-stream, one "data: i <time>" every S seconds
  GET  anything else        -> 200 {"from": <node hostname>, "path": <path>}
  POST anything             -> echoes the body
"""

import http.server
import json
import socket
import sys
import time


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A002
        print(f"[stub {time.strftime('%H:%M:%S')}] {self.client_address[0]} {fmt % args}", flush=True)

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path, _, qs = self.path.partition("?")
        if path.endswith("/stream"):
            q = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
            n, dt = int(q.get("n", "10")), float(q.get("dt", "1"))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(n):
                chunk = f"data: {i} {time.time():.3f}\n\n".encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
                time.sleep(dt)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        self._send(200, json.dumps({"from": socket.gethostname(), "path": path}).encode())

    def do_POST(self):  # noqa: N802
        data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self._send(200, data, "application/octet-stream")


if __name__ == "__main__":
    port = int(sys.argv[1])
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[stub {time.strftime('%H:%M:%S')}] listening on 127.0.0.1:{port} ({socket.gethostname()})", flush=True)
    server.serve_forever()
