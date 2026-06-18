#!/usr/bin/env python3
"""Minimal HTTP CONNECT proxy for testing --remote_proxy=http://host:port."""

import select
import socket
import sys
import threading

_log_lock = threading.Lock()
_log_path = None
# When set, every CONNECT is dialed to this host instead of the requested one. Lets a test target
# the proxy with an unresolvable hostname (proving the client did not resolve it locally) while the
# proxy still reaches the real backend.
_force_host = None


def record_connect(target):
    """Append a CONNECT target to the log file so tests can assert proxy usage."""
    if _log_path is None:
        return
    with _log_lock:
        with open(_log_path, "a") as f:
            f.write(target + "\n")


def handle_connect(client_sock):
    """Handle one HTTP CONNECT request."""
    try:
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = client_sock.recv(4096)
            if not chunk:
                return
            data += chunk

        request_line = data.split(b"\r\n")[0].decode()
        # e.g. "CONNECT buildfarm-server:8980 HTTP/1.1"
        parts = request_line.split()
        if len(parts) < 2 or parts[0] != "CONNECT":
            client_sock.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            return

        target = parts[1]
        record_connect(target)
        host, port = target.rsplit(":", 1)
        port = int(port)
        if _force_host is not None:
            host = _force_host

        try:
            remote_sock = socket.create_connection((host, port), timeout=10)
        except Exception as e:
            msg = f"HTTP/1.1 502 Bad Gateway\r\n\r\n{e}"
            client_sock.sendall(msg.encode())
            return

        client_sock.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")

        relay(client_sock, remote_sock)
    except Exception:
        pass
    finally:
        client_sock.close()


def relay(sock_a, sock_b):
    """Bidirectional relay between two sockets."""
    sockets = [sock_a, sock_b]
    try:
        while True:
            readable, _, _ = select.select(sockets, [], [], 30)
            if not readable:
                break
            for s in readable:
                other = sock_b if s is sock_a else sock_a
                data = s.recv(65536)
                if not data:
                    return
                other.sendall(data)
    finally:
        sock_a.close()
        sock_b.close()


def main():
    global _log_path, _force_host
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3128
    if len(sys.argv) > 2:
        _log_path = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3]:
        _force_host = sys.argv[3]
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(64)
    print(f"HTTP CONNECT proxy listening on 127.0.0.1:{port}", flush=True)
    while True:
        client, addr = srv.accept()
        t = threading.Thread(target=handle_connect, args=(client,), daemon=True)
        t.start()


if __name__ == "__main__":
    main()
