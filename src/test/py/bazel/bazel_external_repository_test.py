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

import http.server
import os
import socketserver
import threading
from absl.testing import absltest
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
    http_handler = http.server.SimpleHTTPRequestHandler
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
    for f in [
        'six-1.10.0.tar.gz',
        'archive_with_symlink.zip',
        'archive_with_symlink.tar.gz',
        'sparse_archive.tar',
    ]:
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
    rule_definition.extend(self.GetDefaultRepoRules())
    self.ScratchFile('WORKSPACE', rule_definition)
    self.ScratchFile('BUILD')
    self.ScratchFile('third_party/BUILD')
    self.ScratchFile('third_party/six.BUILD', build_file)

    self.RunBazel(['build', '@six_archive//...'])

    fetching_disabled_msg = 'fetching is disabled'

    # Changing the mtime of the BUILD file shouldn't invalidate it.
    os.utime(self.Path('third_party/six.BUILD'), (100, 200))
    _, _, stderr = self.RunBazel(['build', '--nofetch', '@six_archive//...'])
    self.assertNotIn(fetching_disabled_msg, os.linesep.join(stderr))

    # Check that --nofetch prints a warning if the BUILD file is changed.
    self.ScratchFile('third_party/six.BUILD', build_file + ['"a noop string"'])
    _, _, stderr = self.RunBazel(['build', '--nofetch', '@six_archive//...'])
    self.assertIn(fetching_disabled_msg, os.linesep.join(stderr))

    # Test repository reloading after BUILD file changes.
    self.ScratchFile('third_party/six.BUILD', build_file + ['foobar'])
    exit_code, _, stderr = self.RunBazel(
        ['build', '@six_archive//...'], allow_failure=True
    )
    self.assertEqual(exit_code, 1, os.linesep.join(stderr))
    self.assertIn('name \'foobar\' is not defined', os.linesep.join(stderr))

  def testNewHttpZipArchiveWithSymlinks(self):
    ip, port = self._http_server.server_address
    rule_definition = [
        'load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")',
        'http_archive(',
        '    name = "archive_with_symlink",',
        '    urls = ["http://%s:%s/archive_with_symlink.zip"],' % (ip, port),
        '    build_file = "@//:archive_with_symlink.BUILD",',
        '    sha256 = ',
        '  "c9c32a48ff65f6319885246b1bfc704e60dd72fb0405dfafdffe403421a4c83a",'
        ')',
    ]
    rule_definition.extend(self.GetDefaultRepoRules())
    self.ScratchFile('WORKSPACE', rule_definition)
    # In the archive, A is a symlink pointing to B
    self.ScratchFile('archive_with_symlink.BUILD', [
        'filegroup(',
        '    name = "file-A",',
        '    srcs = ["A"],',
        ')',
    ])
    self.ScratchFile('BUILD')
    self.RunBazel([
        'build',
        '@archive_with_symlink//:file-A',
    ])

  def testNewHttpTarArchiveWithSymlinks(self):
    ip, port = self._http_server.server_address
    rule_definition = [
        'load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")',
        'http_archive(',
        '    name = "archive_with_symlink",',
        '    urls = ["http://%s:%s/archive_with_symlink.tar.gz"],' % (ip, port),
        '    build_file = "@//:archive_with_symlink.BUILD",',
        '    sha256 = ',
        '  "5ea20285db1b18134e4efe608e41215687f03cd3e3fdd7529860b175fc12fe76",'
        ')',
    ]
    rule_definition.extend(self.GetDefaultRepoRules())
    self.ScratchFile('WORKSPACE', rule_definition)
    # In the archive, A is a symlink pointing to B
    self.ScratchFile('archive_with_symlink.BUILD', [
        'filegroup(',
        '    name = "file-A",',
        '    srcs = ["A"],',
        ')',
    ])
    self.ScratchFile('BUILD')
    self.RunBazel([
        'build',
        '@archive_with_symlink//:file-A',
    ])

  def testNewHttpTarWithSparseFile(self):
    # Test extraction of tar archives containing sparse files.
    # The archive under test was produced using GNU tar on Linux:
    #   truncate -s 1M sparse_file
    #   tar -c --sparse --format posix -f sparse_archive.tar sparse_file

    ip, port = self._http_server.server_address
    rule_definition = [
        'load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")',
        'http_archive(',
        '    name = "sparse_archive",',
        '    urls = ["http://%s:%s/sparse_archive.tar"],' % (ip, port),
        '    build_file = "@//:sparse_archive.BUILD",',
        '    sha256 = ',
        (
            '  "a1a2b2ce4acd51a8cc1ab80adce6f134ac73e885219911a960a42000e312bb65",'
            ')'
        ),
    ]
    rule_definition.extend(self.GetDefaultRepoRules())
    self.ScratchFile('WORKSPACE', rule_definition)
    self.ScratchFile(
        'sparse_archive.BUILD',
        [
            'exports_files(["sparse_file"])',
        ],
    )
    self.ScratchFile(
        'test.py',
        [
            'import sys',
            'from tools.python.runfiles import runfiles',
            'path = runfiles.Create().Rlocation(sys.argv[1])',
            'if open(path, "rb").read() != b"\\0"*1024*1024:',
            '  sys.exit(1)',
        ],
    )
    self.ScratchFile(
        'BUILD',
        [
            'py_test(',
            '    name = "test",',
            '    srcs = ["test.py"],',
            (
                '    args = ["$(rlocationpath @sparse_archive//:sparse_file)"],'
                '    data = ['
            ),
            '        "@sparse_archive//:sparse_file",',
            '        "@bazel_tools//tools/python/runfiles",',
            '    ],)',
        ],
    )
    self.RunBazel([
        'test',
        '//:test',
    ])

  def _CreatePyWritingStarlarkRule(self, print_string):
    self.ScratchFile('repo/foo.bzl', [
        'def _impl(ctx):',
        '  ctx.actions.write(',
        '      output = ctx.outputs.out,',
        '      content = """from __future__ import print_function',
        'print("%s")""",' % print_string,
        '  )',
        '  return [DefaultInfo(files = depset(direct = [ctx.outputs.out]))]',
        '',
        'gen_py = rule(',
        '    implementation = _impl,',
        "    outputs = {'out': '%{name}.py'},",
        ')',
    ])

  def testNewLocalRepositoryNoticesFileChangeInRepoRoot(self):
    """Regression test for https://github.com/bazelbuild/bazel/issues/7063."""
    rule_definition = [
        'load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")',
        (
            'load("@bazel_tools//tools/build_defs/repo:local.bzl",'
            ' "new_local_repository")'
        ),
        'new_local_repository(',
        '    name = "r",',
        '    path = "./repo",',
        '    build_file_content = "exports_files([\'foo.bzl\'])",',
        ')',
    ]
    rule_definition.extend(self.GetDefaultRepoRules())
    self.ScratchFile('WORKSPACE', rule_definition)
    self.CreateWorkspaceWithDefaultRepos('repo/WORKSPACE')
    self._CreatePyWritingStarlarkRule('hello!')
    self.ScratchFile('BUILD', [
        'load("@r//:foo.bzl", "gen_py")',
        'gen_py(name = "gen")',
        'py_binary(name = "bin", srcs = [":gen"], main = "gen.py")',
    ])

    _, stdout, _ = self.RunBazel(['run', '//:bin'])
    self.assertIn('hello!', os.linesep.join(stdout))

    # Modify the definition of the Starlark rule in the external repository.
    # The py_binary rule should notice this and rebuild.
    self._CreatePyWritingStarlarkRule('world')
    _, stdout, _ = self.RunBazel(['run', '//:bin'])
    self.assertNotIn('hello!', os.linesep.join(stdout))
    self.assertIn('world', os.linesep.join(stdout))

  def testDeletedPackagesOnExternalRepo(self):
    self.ScratchFile('other_repo/WORKSPACE')
    self.ScratchFile('other_repo/pkg/BUILD', [
        'filegroup(',
        '  name = "file",',
        '  srcs = ["ignore/file"],',
        ')',
    ])
    self.ScratchFile('other_repo/pkg/ignore/BUILD', [
        'Bad BUILD file',
    ])
    self.ScratchFile('other_repo/pkg/ignore/file')
    work_dir = self.ScratchDir('my_repo')
    self.ScratchFile(
        'my_repo/WORKSPACE',
        [
            (
                'load("@bazel_tools//tools/build_defs/repo:local.bzl",'
                ' "local_repository")'
            ),
            "local_repository(name = 'other_repo', path='../other_repo')",
        ],
    )

    exit_code, _, stderr = self.RunBazel(
        args=['build', '@other_repo//pkg:file'],
        cwd=work_dir,
        allow_failure=True,
    )
    self.AssertExitCode(exit_code, 1, stderr)
    self.assertIn("'@@other_repo//pkg/ignore' is a subpackage", ''.join(stderr))

    self.RunBazel(
        args=[
            'build',
            '@other_repo//pkg:file',
            '--deleted_packages=@other_repo//pkg/ignore',
        ],
        cwd=work_dir,
    )

  def testBazelignoreFileOnExternalRepo(self):
    self.ScratchFile('other_repo/WORKSPACE')
    self.ScratchFile('other_repo/pkg/BUILD', [
        'filegroup(',
        '  name = "file",',
        '  srcs = ["ignore/file.txt"],',
        ')',
    ])
    self.ScratchFile('other_repo/pkg/ignore/BUILD', [
        'filegroup(',
        '  name = "file",',
        '  srcs = ["file.txt"],',
        ')',
    ])
    self.ScratchFile('other_repo/pkg/ignore/file.txt')
    work_dir = self.ScratchDir('my_repo')
    self.ScratchFile(
        'my_repo/WORKSPACE',
        [
            (
                'load("@bazel_tools//tools/build_defs/repo:local.bzl",'
                ' "local_repository")'
            ),
            'local_repository(name = "other_repo", path="../other_repo")',
        ],
    )

    exit_code, _, stderr = self.RunBazel(
        args=['build', '@other_repo//pkg:file'],
        cwd=work_dir,
        allow_failure=True,
    )
    self.AssertExitCode(exit_code, 1, stderr)
    self.assertIn("'@@other_repo//pkg/ignore' is a subpackage", ''.join(stderr))

    self.ScratchFile('other_repo/.bazelignore', [
        'pkg/ignore',
    ])

    self.RunBazel(args=['build', '@other_repo//pkg:file'], cwd=work_dir)

    self.ScratchFile('my_repo/BUILD', [
        'filegroup(',
        '  name = "all_files",',
        '  srcs = ["@other_repo//pkg/ignore:file"],',
        ')',
    ])
    exit_code, _, stderr = self.RunBazel(
        args=['build', '//:all_files'], cwd=work_dir, allow_failure=True
    )
    self.AssertExitCode(exit_code, 1, stderr)
    self.assertIn("no such package '@@other_repo//pkg/ignore'", ''.join(stderr))

  def testUniverseScopeWithBazelIgnoreInExternalRepo(self):
    self.ScratchFile('other_repo/WORKSPACE')
    self.ScratchFile('other_repo/pkg/BUILD', [
        'filegroup(',
        '  name = "file",',
        '  srcs = ["ignore/file.txt"],',
        ')',
    ])
    # This BUILD file should be ignored.
    self.ScratchFile('other_repo/pkg/ignore/BUILD', [
        'filegroup(',
        '  name = "file",',
        '  srcs = ["file.txt"],',
        ')',
    ])
    self.ScratchFile('other_repo/pkg/ignore/file.txt')
    self.ScratchFile('other_repo/.bazelignore', [
        'pkg/ignore',
    ])

    work_dir = self.ScratchDir('my_repo')
    self.ScratchFile(
        'my_repo/WORKSPACE',
        [
            (
                'load("@bazel_tools//tools/build_defs/repo:local.bzl",'
                ' "local_repository")'
            ),
            'local_repository(name = "other_repo", path="../other_repo")',
        ],
    )

    _, stdout, _ = self.RunBazel(
        args=[
            'query',
            '--universe_scope=@other_repo//...:*',
            '--order_output=no',
            'deps(@other_repo//pkg:file)',
        ],
        cwd=work_dir,
    )
    self.assertIn('@other_repo//pkg:ignore/file.txt', ''.join(stdout))

  def testBazelignoreFileFromMainRepoDoesNotAffectExternalRepos(self):
    # Regression test for https://github.com/bazelbuild/bazel/issues/10234
    self.ScratchFile('other_repo/WORKSPACE')
    self.ScratchFile('other_repo/foo/bar/BUILD', [
        'filegroup(',
        '  name = "file",',
        '  srcs = ["file.txt"],',
        ')',
    ])
    self.ScratchFile('other_repo/foo/bar/file.txt')

    work_dir = self.ScratchDir('my_repo')
    self.ScratchFile(
        'my_repo/WORKSPACE',
        [
            (
                'load("@bazel_tools//tools/build_defs/repo:local.bzl",'
                ' "local_repository")'
            ),
            'local_repository(name = "other_repo", path="../other_repo")',
        ],
    )
    # This should not exclude @other_repo//foo/bar
    self.ScratchFile('my_repo/.bazelignore', ['foo/bar'])

    _, stdout, _ = self.RunBazel(
        args=['query', '@other_repo//foo/bar/...'], cwd=work_dir
    )
    self.assertIn('@other_repo//foo/bar:file', ''.join(stdout))

  def testBazelignoreFileFromExternalRepoDoesNotAffectMainRepo(self):
    self.ScratchFile('other_repo/WORKSPACE')
    # This should not exclude //foo/bar in main repo
    self.ScratchFile('other_repo/.bazelignore', ['foo/bar'])
    self.ScratchFile('other_repo/BUILD',)

    work_dir = self.ScratchDir('my_repo')
    self.ScratchFile('my_repo/foo/bar/BUILD', [
        'filegroup(',
        '  name = "file",',
        '  srcs = ["file.txt"],',
        ')',
    ])
    self.ScratchFile('my_repo/foo/bar/file.txt')
    self.ScratchFile(
        'my_repo/WORKSPACE',
        [
            (
                'load("@bazel_tools//tools/build_defs/repo:local.bzl",'
                ' "local_repository")'
            ),
            'local_repository(name = "other_repo", path="../other_repo")',
        ],
    )

    _, stdout, _ = self.RunBazel(args=['query', '//foo/bar/...'], cwd=work_dir)
    self.assertIn('//foo/bar:file', ''.join(stdout))

  def testMainBazelignoreContainingRepoName(self):
    self.ScratchFile('other_repo/WORKSPACE')
    self.ScratchFile('other_repo/foo/bar/BUILD', [
        'filegroup(',
        '  name = "file",',
        '  srcs = ["file.txt"],',
        ')',
    ])
    self.ScratchFile('other_repo/foo/bar/file.txt')

    work_dir = self.ScratchDir('my_repo')
    self.ScratchFile(
        'my_repo/WORKSPACE',
        [
            (
                'load("@bazel_tools//tools/build_defs/repo:local.bzl",'
                ' "local_repository")'
            ),
            'local_repository(name = "other_repo", path="../other_repo")',
        ],
    )
    # This should not exclude @other_repo//foo/bar, because .bazelignore doesn't
    # support having repository name in the path fragment.
    self.ScratchFile('my_repo/.bazelignore', ['@other_repo//foo/bar'])

    self.RunBazel(args=['build', '@other_repo//foo/bar:file'], cwd=work_dir)

  def testExternalBazelignoreContainingRepoName(self):
    self.ScratchFile('other_repo/WORKSPACE')
    # This should not exclude @third_repo//foo/bar, because .bazelignore doesn't
    # support having repository name in the path fragment.
    self.ScratchFile('other_repo/.bazelignore', ['@third_repo//foo/bar'])
    self.ScratchFile('other_repo/BUILD', [
        'filegroup(',
        '  name = "file",',
        '  srcs = ["@third_repo//foo/bar:file"],',
        ')',
    ])

    self.ScratchFile('third_repo/WORKSPACE')
    self.ScratchFile('third_repo/foo/bar/BUILD', [
        'filegroup(',
        '  name = "file",',
        '  srcs = ["file.txt"],',
        '  visibility = ["//visibility:public"],',
        ')',
    ])
    self.ScratchFile('third_repo/foo/bar/file.txt')

    work_dir = self.ScratchDir('my_repo')
    self.ScratchFile(
        'my_repo/WORKSPACE',
        [
            (
                'load("@bazel_tools//tools/build_defs/repo:local.bzl",'
                ' "local_repository")'
            ),
            'local_repository(name = "other_repo", path="../other_repo")',
            'local_repository(name = "third_repo", path="../third_repo")',
        ],
    )

    self.RunBazel(args=['build', '@other_repo//:file'], cwd=work_dir)

  def testRepoEnv(self):
    # Testing fix for issue: https://github.com/bazelbuild/bazel/issues/15430

    self.ScratchFile('WORKSPACE', [
        'load("//:main.bzl", "dump_env")', 'dump_env(', '    name = "debug"',
        ')'
    ])
    self.ScratchFile('BUILD')
    self.ScratchFile('main.bzl', [
        'def _dump_env(ctx):', '    val = ctx.os.environ.get("FOO", "nothing")',
        '    print("FOO", val)', '    ctx.file("BUILD")',
        'dump_env = repository_rule(', '    implementation = _dump_env,',
        '    local = True,', ')'
    ])

    _, _, stderr = self.RunBazel(
        args=['build', '@debug//:all', '--repo_env=FOO'], env_add={'FOO': 'bar'}
    )
    self.assertIn('FOO bar', os.linesep.join(stderr))
    self.assertNotIn('null value in entry: FOO=null', os.linesep.join(stderr))

if __name__ == '__main__':
  absltest.main()
