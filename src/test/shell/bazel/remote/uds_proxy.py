# Copyright 2019 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple replaying Unix Domain Socket (UDS) server."""

import contextlib
import select
import socket
import sys
import threading


def main(argv=None):
  if argv is None:
    argv = sys.argv[1:]

  srv_address = argv[0]
  dst_host, dst_port = argv[1].split(":")
  dst_address = (dst_host, int(dst_port))

  srv_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  srv_sock.bind(srv_address)
  srv_sock.listen(64)
  while True:
    src_sock, _ = srv_sock.accept()
    proxy_thread = threading.Thread(target=proxy, args=[src_sock, dst_address])
    proxy_thread.daemon = True
    proxy_thread.start()


def proxy(src_sock, dst_address):
  """Read data from a USD socket and write it back."""
  with contextlib.closing(src_sock):
    src_sock.settimeout(None)
    dst_sock = socket.create_connection(dst_address)
    with contextlib.closing(dst_sock):
      dst_sock.settimeout(None)

      while True:
        readable, _, _ = select.select([src_sock, dst_sock], [], [])
        if src_sock in readable:
          data = src_sock.recv(4096)
          if not data:
            return
          dst_sock.sendall(data)
        if dst_sock in readable:
          data = dst_sock.recv(4096)
          if data:
            src_sock.sendall(data)


if __name__ == "__main__":
  sys.exit(main())
