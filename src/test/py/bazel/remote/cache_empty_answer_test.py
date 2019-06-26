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

import os
import shutil
import socket
import stat
import threading
import unittest
from src.test.py.bazel import test_base

try:
  from BaseHTTPServer import BaseHTTPRequestHandler
except ImportError:
  from http.server import BaseHTTPRequestHandler

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
    self.finish()

  def do_GET(self):
    self.send_response(200)
    self.send_header('Content-Length', '0')
    self.end_headers()
    self.finish()

class CacheEmptyAnswerTest(test_base.TestBase):

  def setUp(self):
    test_base.TestBase.setUp(self)
    server_port = self.GetFreeTCPPort()
    self.httpd = HTTPServer(('localhost', server_port), MemoryStorageHandler)
    self.httpd.storage = {}
    self.url = 'http://localhost:{}'.format(server_port)
    self.background = threading.Thread(target=self.httpd.serve_forever)
    self.background.start()

  def tearDown(self):
    self.httpd.shutdown()
    self.background.join()
    test_base.TestBase.tearDown(self)

  def testEmptyAnswerTreatedAsMiss(self):
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

    exit_code, _, stderr = self.RunBazel(['build', '//:genrule.txt', '--remote_http_cache', self.url])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertNotIn('INFO: 1 process: 1 remote cache hit.', stderr)

  def testEmptyAnswerWithNoOutputsTreatedAsHit(self):
    content = ['Hello HTTP']
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('input.txt', content)
    self.ScratchFile('myrule.bzl', [
        'def _impl(ctx):',
        '    out = ctx.actions.declare_file("myrule.txt")',
        '    ctx.actions.expand_template(template=ctx.file.src, output=out, substitutions={})',
        '    return [DefaultInfo(files=depset([out]))]',
        '',
        'myrule = rule(',
        '    implementation=_impl,',
        '    attrs={',
        '        "src": attr.label(allow_single_file=True),',
        '    },'
        ')'
    ])
    self.ScratchFile('BUILD', [
        'load(":myrule.bzl", "myrule")',
        '',
        'myrule(',
        '    name = "myrule",',
        '    src = "//:input.txt",',
        ')',
    ])

    exit_code, _, stderr = self.RunBazel(['build', '//:myrule', '--remote_http_cache', self.url])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn('INFO: 1 process: 1 remote cache hit.', stderr)

  def GetFreeTCPPort(self):
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('localhost', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port

  def AssertFileContentEqual(self, file_path, entry):
    with open(file_path, 'r') as f:
      self.assertEqual(f.read().splitlines(), entry)


if __name__ == '__main__':
  unittest.main()
