"""TCP listener for the reachability test: logs every connection (peer address, first line sent) and
answers with one line. Runs until killed.

    python3 listen.py <port>
"""

import socket
import sys
import threading
import time


def log(msg: str) -> None:
    print(f"[listen {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def handle(conn: socket.socket, peer: tuple) -> None:
    with conn:
        conn.settimeout(5)
        try:
            data = conn.recv(256)
        except OSError as e:
            data = f"<recv failed: {e}>".encode()
        log(f"CONNECTION from {peer[0]}:{peer[1]} sent {data[:120]!r}")
        try:
            conn.sendall(f"pong from {socket.gethostname()}\n".encode())
        except OSError:
            pass


def main() -> None:
    port = int(sys.argv[1])
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(16)
    log(f"listening on 0.0.0.0:{port} ({socket.gethostname()})")
    while True:
        conn, peer = srv.accept()
        threading.Thread(target=handle, args=(conn, peer), daemon=True).start()


if __name__ == "__main__":
    main()
