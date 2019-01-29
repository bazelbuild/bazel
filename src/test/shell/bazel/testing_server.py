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
import base64
try:
  from http.server import BaseHTTPRequestHandler
except ImportError:
  # Python 2.x compatibility hack.
  from BaseHTTPServer import BaseHTTPRequestHandler
import os
import os.path
try:
  from socketserver import TCPServer
except ImportError:
  # Python 2.x compatibility hack.
  from SocketServer import TCPServer
import sys


class Handler(BaseHTTPRequestHandler):
  """Handlers for testing HTTP server."""
  auth = False
  valid_header = b'Basic ' + base64.b64encode('foo:bar'.encode('ascii'))

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
    if not self.auth:
      self.do_HEAD()
      self.serve_file()
      return

    auth_header = self.headers.get('Authorization', '').encode('ascii')
    if auth_header == self.valid_header:
      self.do_HEAD()
      self.serve_file()
    else:
      self.do_AUTHHEAD()
      self.wfile.write(b'Login required.')

  def serve_file(self):
    with open(os.path.join(os.getcwd(), self.path[1:]), 'rb') as file_to_serve:
      self.wfile.write(file_to_serve.read())


def main(argv=None):
  if argv is None:
    argv = sys.argv[1:]

  if not argv:
    sys.stderr.write('Usage: testing_server.py port [auth]\n')
    return 1

  port = int(argv[0])
  if len(argv) > 1:
    Handler.auth = True

  httpd = TCPServer(('', port), Handler)

  try:
    sys.stderr.write('Serving forever on %d.\n' % port)
    httpd.serve_forever()
  finally:
    sys.stderr.write('Goodbye.\n')


if __name__ == '__main__':
  sys.exit(main())
