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

import base64
from BaseHTTPServer import BaseHTTPRequestHandler
import getopt
import os
import SocketServer
import sys

auth = None


class Handler(BaseHTTPRequestHandler):
  """Handlers for testing HTTP server."""

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
    if auth is None:
      self.do_HEAD()
      self.serve_file()
      return

    foo_bar = base64.b64encode('foo:bar')

    if self.headers.getheader('Authorization') is None:
      self.do_AUTHHEAD()
      self.wfile.write('Login required.')
    elif self.headers.getheader('Authorization') == 'Basic %s' % foo_bar:
      self.do_HEAD()
      try:
        self.wfile.write(self.headers.getheader('Authorization'))
        self.serve_file()
      except IOError:
        self.wfile.write('Authorized.')
    else:
      self.do_AUTHHEAD()
      self.wfile.write(self.headers.getheader('Authorization'))
      self.wfile.write('not authenticated')

  def serve_file(self):
    file_to_serve = open(str(os.getcwd()) + self.path)
    self.wfile.write(file_to_serve.read())


if __name__ == '__main__':
  try:
    opts, args = getopt.getopt(sys.argv[1:], 'p:a:', ['port=', 'auth='])
  except getopt.GetoptError:
    print 'Error parsing args'
    sys.exit(1)

  port = 12345
  for o, a in opts:
    if o in ('-p', '--port'):
      port = int(a)
    if o in ('-a', '--auth'):
      auth = a
  httpd = SocketServer.TCPServer(('', port), Handler)
  try:
    print 'Serving forever on %d.' % port
    httpd.serve_forever()
  except:  # pylint: disable=bare-except
    print 'Goodbye.'
