# Copyright 2017 The Bazel Authors. All rights reserved.
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

import gzip
import io
import os
import socket
import threading
from absl.testing import absltest
from src.test.py.bazel import test_base

# pylint: disable=g-import-not-at-top,g-importing-member
try:
  from BaseHTTPServer import BaseHTTPRequestHandler
except ImportError:
  from http.server import BaseHTTPRequestHandler

# pylint: disable=g-import-not-at-top,g-importing-member
try:
  from BaseHTTPServer import HTTPServer
except ImportError:
  from http.server import HTTPServer


class MemoryStorageHandler(BaseHTTPRequestHandler):
  protocol_version = 'HTTP/1.1'

  def __init__(self, request, client_address, server):
    BaseHTTPRequestHandler.__init__(self, request, client_address, server)

  def do_PUT(self):
    self.send_response(200)
    self.end_headers()
    self.server.storage[self.path] = self.rfile.read(
        int(self.headers['Content-Length']))
    self.finish()

  def do_GET(self):
    if self.path in self.server.storage and 'gzip' in self.headers[
        'Accept-Encoding']:
      out = io.BytesIO()
      with gzip.GzipFile(fileobj=out, mode='w') as f:
        f.write(self.server.storage[self.path])
      self.send_response(200)
      self.send_header('Content-Length', str(len(out.getvalue())))
      self.send_header('Content-Encoding', 'gzip')
      self.end_headers()
      self.wfile.write(out.getvalue())

    else:
      self.send_response(404)
      self.end_headers()
    self.finish()


class HTTPServerV6(HTTPServer):
  address_family = socket.AF_INET6


class CacheDecompressionTest(test_base.TestBase):

  def setUp(self):
    test_base.TestBase.setUp(self)
    server_port = self.GetFreeTCPPort()
    if self.IsDarwin():
      self.httpd = HTTPServerV6(
          ('localhost', server_port), MemoryStorageHandler
      )
    else:
      self.httpd = HTTPServer(('localhost', server_port), MemoryStorageHandler)
    self.httpd.storage = {}
    self.url = 'http://localhost:{}'.format(server_port)
    self.background = threading.Thread(target=self.httpd.serve_forever)
    self.background.start()

  def tearDown(self):
    self.httpd.shutdown()
    self.background.join()
    test_base.TestBase.tearDown(self)

  def testDecompressionWorks(self):
    content = ['Hello HTTP']
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('input.txt', content)
    self.ScratchFile('BUILD', [
        'genrule(',
        '    name = "genrule",',
        '    cmd = "cp \\"$<\\" \\"$@\\"",',
        '    srcs = ["//:input.txt"],',
        '    outs = ["genrule.txt"],',
        ')',
    ])

    _, _, stderr = self.RunBazel(
        ['build', '//:genrule.txt', '--remote_cache', self.url]
    )
    self.assertNotIn('INFO: 2 processes: 1 remote cache hit, 1 internal',
                     stderr)
    self.assertNotIn('HTTP version 1.1 is required', stderr)

    self.RunBazel(['clean', '--expunge'])

    _, _, stderr = self.RunBazel(
        ['build', '//:genrule.txt', '--remote_cache', self.url]
    )
    self.assertIn('INFO: 2 processes: 1 remote cache hit, 1 internal.', stderr)
    self.assertNotIn('HTTP version 1.1 is required', stderr)

    _, stdout, _ = self.RunBazel(['info', 'bazel-genfiles'])
    bazel_genfiles = stdout[0]

    self.AssertFileContentEqual(
        os.path.join(bazel_genfiles, 'genrule.txt'), content)

  def GetFreeTCPPort(self):
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('localhost', 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port

  def AssertFileContentEqual(self, file_path, entry):
    with open(file_path, 'r') as f:
      self.assertEqual(f.read().splitlines(), entry)


if __name__ == '__main__':
  absltest.main()
