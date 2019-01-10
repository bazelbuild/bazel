# pylint: disable=g-bad-file-header
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
import threading
import unittest
from six.moves import SimpleHTTPServer
from six.moves import socketserver
from src.test.py.bazel import test_base


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
  """A helper class to launcher a threaded http server."""
  pass


class BazelExternalRepositoryTest(test_base.TestBase):

  _http_server = None

  def StartHttpServer(self):
    """Runs a simple http server to serve files under current directory."""
    # Port 0 means to select an arbitrary unused port
    host, port = 'localhost', 0
    http_handler = SimpleHTTPServer.SimpleHTTPRequestHandler
    server = ThreadedTCPServer((host, port), http_handler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    self._http_server = server

  def StopHttpServer(self):
    """Shutdown and clean up the http server."""
    if self._http_server:
      self._http_server.shutdown()
      self._http_server.server_close()

  def setUp(self):
    test_base.TestBase.setUp(self)
    for f in ['six-1.10.0.tar.gz', 'archive_with_symlink.zip']:
      self.CopyFile(self.Rlocation('io_bazel/src/test/py/bazel/testdata/'
                                   'bazel_external_repository_test/' + f), f)
    self.StartHttpServer()

  def tearDown(self):
    test_base.TestBase.tearDown(self)
    self.StopHttpServer()

  def testNewHttpArchive(self):
    ip, port = self._http_server.server_address
    rule_definition = [
        'load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")',
        'http_archive(',
        '    name = "six_archive",',
        '    urls = ["http://%s:%s/six-1.10.0.tar.gz"],' % (ip, port),
        '    sha256 = '
        '"105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",',
        '    strip_prefix = "six-1.10.0",',
        '    build_file = "@//third_party:six.BUILD",',
        ')',
    ]
    build_file = [
        'py_library(',
        '  name = "six",',
        '  srcs = ["six.py"],',
        ')',
    ]
    self.ScratchFile('WORKSPACE', rule_definition)
    self.ScratchFile('BUILD')
    self.ScratchFile('third_party/BUILD')
    self.ScratchFile('third_party/six.BUILD', build_file)

    exit_code, _, stderr = self.RunBazel(['build', '@six_archive//...'])
    self.assertEqual(exit_code, 0, os.linesep.join(stderr))

    fetching_disabled_msg = 'fetching is disabled'

    # Changing the mtime of the BUILD file shouldn't invalidate it.
    os.utime(self.Path('third_party/six.BUILD'), (100, 200))
    exit_code, _, stderr = self.RunBazel(
        ['build', '--nofetch', '@six_archive//...'])
    self.assertEqual(exit_code, 0, os.linesep.join(stderr))
    self.assertNotIn(fetching_disabled_msg, os.linesep.join(stderr))

    # Check that --nofetch prints a warning if the BUILD file is changed.
    self.ScratchFile('third_party/six.BUILD', build_file + ['"a noop string"'])
    exit_code, _, stderr = self.RunBazel(
        ['build', '--nofetch', '@six_archive//...'])
    self.assertEqual(exit_code, 0, os.linesep.join(stderr))
    self.assertIn(fetching_disabled_msg, os.linesep.join(stderr))

    # Test repository reloading after BUILD file changes.
    self.ScratchFile('third_party/six.BUILD', build_file + ['foobar'])
    exit_code, _, stderr = self.RunBazel(['build', '@six_archive//...'])
    self.assertEqual(exit_code, 1, os.linesep.join(stderr))
    self.assertIn('name \'foobar\' is not defined', os.linesep.join(stderr))

  def testNewHttpArchiveWithSymlinks(self):
    ip, port = self._http_server.server_address
    self.ScratchFile('WORKSPACE', [
        'load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")',
        'http_archive(',
        '    name = "archive_with_symlink",',
        '    urls = ["http://%s:%s/archive_with_symlink.zip"],' % (ip, port),
        '    build_file = "@//:archive_with_symlink.BUILD",',
        ')',
    ])
    # In the archive, A is a symlink pointing to B
    self.ScratchFile('archive_with_symlink.BUILD', [
        'filegroup(',
        '    name = "file-A",',
        '    srcs = ["A"],',
        ')',
    ])
    self.ScratchFile('BUILD')
    exit_code, _, stderr = self.RunBazel([
        'build',
        '@archive_with_symlink//:file-A',
    ])
    self.assertEqual(exit_code, 0, os.linesep.join(stderr))

  def _CreatePyWritingStarlarkRule(self, print_string):
    self.ScratchFile('repo/foo.bzl', [
        "def _impl(ctx):",
        "  ctx.actions.write(",
        "      output = ctx.outputs.out,",
        "      content = \"\"\"from __future__ import print_function",
        "print('%s')\"\"\"," % print_string,
        "  )",
        "  return [DefaultInfo(files = depset(direct = [ctx.outputs.out]))]",
        "",
        "gen_py = rule(",
        "    implementation = _impl,",
        "    outputs = {'out': '%{name}.py'},",
        ")",
    ])

  def testNewLocalRepositoryNoticesFileChangeInRepoRoot(self):
    """Regression test for https://github.com/bazelbuild/bazel/issues/7063."""
    self.ScratchFile('WORKSPACE', [
        'new_local_repository(',
        '    name = "r",',
        '    path = "./repo",',
        '    build_file_content = "exports_files([\'foo.bzl\'])",',
        ')',
    ])
    self.ScratchFile('repo/WORKSPACE')
    self._CreatePyWritingStarlarkRule('hello!')
    self.ScratchFile('BUILD', [
        'load("@r//:foo.bzl", "gen_py")',
        'gen_py(name = "gen")',
        'py_binary(name = "bin", srcs = [":gen"], main = "gen.py")',
    ])

    exit_code, stdout, stderr = self.RunBazel(['run', '//:bin'])
    self.assertEqual(exit_code, 0, os.linesep.join(stderr))
    self.assertIn('hello!', os.linesep.join(stdout))

    # Modify the definition of the Starlark rule in the external repository.
    # The py_binary rule should notice this and rebuild.
    self._CreatePyWritingStarlarkRule('world')
    exit_code, stdout, stderr = self.RunBazel(['run', '//:bin'])
    self.assertEqual(exit_code, 0, os.linesep.join(stderr))
    self.assertNotIn('hello!', os.linesep.join(stdout))
    self.assertIn('world', os.linesep.join(stdout))

if __name__ == '__main__':
  unittest.main()
