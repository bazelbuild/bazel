# Copyright 2015 The Bazel Authors. All rights reserved.
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

"""An HTTP server to use for external repository integration tests."""

# pylint: disable=g-import-not-at-top,g-importing-member
import argparse
import base64
import json

try:
  from http.server import BaseHTTPRequestHandler
except ImportError:
  # Python 2.x compatibility hack.
  from BaseHTTPServer import BaseHTTPRequestHandler
import os
import os.path
try:
  from socketserver import TCPServer
  if os.name != 'nt':
    from socketserver import UnixStreamServer
except ImportError:
  # Python 2.x compatibility hack.
  from SocketServer import TCPServer
  if os.name != 'nt':
    from SocketServer import UnixStreamServer
import random
import socket
import sys
import time


class TCPServerV6(TCPServer):
  address_family = socket.AF_INET6


class Handler(BaseHTTPRequestHandler):
  """Handlers for testing HTTP server."""
  auth = False
  not_found = False
  simulate_timeout = False
  dump_headers = None
  filename = None
  redirect = None
  valid_headers = [
      b'Basic ' + base64.b64encode('foo:bar'.encode('ascii')), b'Bearer TOKEN'
  ]
  unstable_headers = ['Host', 'User-Agent']

  def do_HEAD(self):  # pylint: disable=invalid-name
    self.send_response(200)
    self.send_header('Content-type', 'text/html')
    self.end_headers()

  def do_AUTHHEAD(self):  # pylint: disable=invalid-name
    self.send_response(401)
    self.send_header('WWW-Authenticate', 'Basic realm=\"Bazel\"')
    self.send_header('Content-type', 'text/html')
    self.end_headers()

  def do_GET(self):  # pylint: disable=invalid-name
    if not self.client_address:
      # Needed for Unix domain connections as the response functions
      # fail without this being set.
      self.client_address = 'localhost'

    if self.dump_headers:
      headers = filter(
          lambda hv: hv[0] not in self.unstable_headers, self.headers.items()
      )
      with open(self.dump_headers, 'w') as f:
        f.write(json.dumps(dict(headers)))

    if self.simulate_timeout:
      while True:
        time.sleep(1)

    if self.not_found:
      self.send_response(404)
      self.end_headers()
      return

    if self.redirect is not None:
      self.send_response(301)
      self.send_header('Location', self.redirect)
      self.end_headers()
      return

    if not self.auth:
      self.do_HEAD()
      self.serve_file()
      return

    auth_header = self.headers.get('Authorization', '').encode('ascii')

    if auth_header in self.valid_headers:
      self.do_HEAD()
      self.serve_file()
    else:
      self.do_AUTHHEAD()
      self.wfile.write(
          'Bad authorization header: {}'.format(auth_header).encode('ascii')
      )

  def serve_file(self):
    path_to_serve = self.path[1:]
    if self.filename is not None:
      path_to_serve = self.filename
    to_serve = os.path.join(os.getcwd(), path_to_serve)
    with open(to_serve, 'rb') as file_to_serve:
      self.wfile.write(file_to_serve.read())


def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--unix_socket', action='store')
  parser.add_argument('--dump_headers', action='store')

  parser.add_argument('mode', type=str, nargs='?')
  parser.add_argument('target', type=str, nargs='?')
  args = parser.parse_args(argv)

  if args.mode:
    if args.mode == 'always' and args.target:
      Handler.filename = args.target
    elif args.mode == 'redirect' and args.target:
      Handler.redirect = args.target
    elif args.mode == '404':
      Handler.not_found = True
    elif args.mode == 'timeout':
      Handler.simulate_timeout = True
    elif args.mode == 'auth':
      Handler.auth = True
      if args.target:
        Handler.filename = args.target

  Handler.dump_headers = args.dump_headers

  httpd = None
  if args.unix_socket:
    httpd = UnixStreamServer(args.unix_socket, Handler)
    sys.stderr.write('Serving forever on %s.\n' % args.unix_socket)
  else:
    port = None
    while port is None:
      try:
        port = random.randrange(32760, 59760)
        if sys.platform == 'darwin':
          httpd = TCPServerV6(('', port), Handler)
        else:
          httpd = TCPServer(('', port), Handler)
      except socket.error:
        port = None
    sys.stdout.write('%d\nstarted\n' % (port,))
    sys.stdout.flush()
    sys.stdout.close()
    sys.stderr.write('Serving forever on %d.\n' % port)

  try:
    httpd.serve_forever()
  finally:
    sys.stderr.write('Goodbye.\n')


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
