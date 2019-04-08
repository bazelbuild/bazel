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
import random
import socket
import sys


class Handler(BaseHTTPRequestHandler):
  """Handlers for testing HTTP server."""
  auth = False
  not_found = False
  filename = None
  redirect = None
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
    if auth_header == self.valid_header:
      self.do_HEAD()
      self.serve_file()
    else:
      self.do_AUTHHEAD()
      self.wfile.write(b'Login required.')

  def serve_file(self):
    path_to_serve = self.path[1:]
    if self.filename is not None:
      path_to_serve = self.filename
    to_serve = os.path.join(os.getcwd(), path_to_serve)
    with open(to_serve, 'rb') as file_to_serve:
      self.wfile.write(file_to_serve.read())


def main(argv=None):
  if argv is None:
    argv = sys.argv[1:]

  if len(argv) > 1 and argv[0] == 'always':
    Handler.filename = argv[1]
  elif len(argv) > 1 and argv[0] == 'redirect':
    Handler.redirect = argv[1]
  elif argv and argv[0] == '404':
    Handler.not_found = True
  elif argv:
    Handler.auth = True

  httpd = None
  port = None
  while port is None:
    try:
      port = random.randrange(32760, 59760)
      httpd = TCPServer(('', port), Handler)
    except socket.error:
      port = None

  try:
    sys.stdout.write('%d\nstarted\n' % (port,))
    sys.stdout.flush()
    sys.stdout.close()
    sys.stderr.write('Serving forever on %d.\n' % port)
    httpd.serve_forever()
  finally:
    sys.stderr.write('Goodbye.\n')


if __name__ == '__main__':
  sys.exit(main())
