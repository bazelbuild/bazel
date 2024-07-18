# Copyright 2018 The Bazel Authors. All rights reserved.
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
import zipfile
from absl.testing import absltest
from src.test.py.bazel import test_base


class PyTest(test_base.TestBase):
  """Integration tests for the Python rules of Bazel."""

  def createSimpleFiles(self):

    self.ScratchFile(
        'a/BUILD',
        [
            'py_binary(name="a", srcs=["a.py"], deps=[":b"])',
            'py_library(name="b", srcs=["b.py"], imports=["."])',
        ],
    )

    self.ScratchFile(
        'a/a.py',
        [
            'import b',
            'b.Hello()',
        ])

    self.ScratchFile(
        'a/b.py',
        [
            'def Hello():',
            '    print("Hello, World")',
        ])

  def testSmoke(self):
    self.createSimpleFiles()
    _, stdout, _ = self.RunBazel(['run', '//a:a'])
    self.assertIn('Hello, World', stdout)

  def testRunfilesSymlinks(self):
    if test_base.TestBase.IsWindows():
      # No runfiles symlinks on Windows
      return

    self.createSimpleFiles()
    self.RunBazel(['build', '//a:a'])
    self.assertTrue(os.path.isdir('bazel-bin/a/a.runfiles'))
    self.assertTrue(
        os.readlink('bazel-bin/a/a.runfiles/_main/a/a.py').endswith('/a/a.py')
    )
    self.assertTrue(
        os.readlink('bazel-bin/a/a.runfiles/_main/a/b.py').endswith('/a/b.py')
    )


class TestInitPyFiles(test_base.TestBase):

  def createSimpleFiles(self, create_init=True):

    self.ScratchFile('src/a/BUILD', [
        'py_binary(name="a", srcs=["a.py"], deps=[":b"], legacy_create_init=%s)'
        % create_init,
        'py_library(name="b", srcs=["b.py"])',
    ])

    self.ScratchFile('src/a/a.py', [
        'from src.a import b',
        'b.Hello()',
    ])

    self.ScratchFile('src/a/b.py', [
        'def Hello():',
        '    print("Hello, World")',
    ])

  def testInitPyFilesCreated(self):
    self.createSimpleFiles()
    self.RunBazel(['build', '//src/a:a'])
    if self.IsWindows():
      # On Windows Bazel creates bazel-bin/src/a/a.zip
      self.assertTrue(os.path.exists('bazel-bin/src/a/a.zip'))
      with zipfile.ZipFile('bazel-bin/src/a/a.zip', 'r') as z:
        zip_contents = set(z.namelist())
      self.assertIn('runfiles/_main/src/__init__.py', zip_contents)
      self.assertIn('runfiles/_main/src/a/__init__.py', zip_contents)
    else:
      self.assertTrue(
          os.path.exists('bazel-bin/src/a/a.runfiles/_main/src/__init__.py')
      )
      self.assertTrue(
          os.path.exists('bazel-bin/src/a/a.runfiles/_main/src/a/__init__.py')
      )

  def testInitPyFilesNotCreatedWhenLegacyCreateInitIsSet(self):
    self.createSimpleFiles(create_init=False)
    self.RunBazel(['build', '//src/a:a'])
    self.assertFalse(
        os.path.exists('bazel-bin/src/a/a.runfiles/_main/src/__init__.py')
    )
    self.assertFalse(
        os.path.exists('bazel-bin/src/a/a.runfiles/_main/src/a/__init__.py')
    )

  # Regression test for https://github.com/bazelbuild/bazel/pull/10119
  def testBuildingZipFileWithTargetNameWithDot(self):
    self.ScratchFile('BUILD', [
        'py_binary(',
        '  name = "bin.v1",  # .v1 should not be treated as extension and removed accidentally',
        '  srcs = ["bin.py"],',
        '  main = "bin.py",',
        ')',
    ])
    self.ScratchFile('bin.py', ['print("Hello, world")'])
    self.RunBazel(['build', '--build_python_zip', '//:bin.v1'])
    self.assertTrue(os.path.exists('bazel-bin/bin.v1.temp'))
    self.assertTrue(os.path.exists('bazel-bin/bin.v1.zip'))


@absltest.skipIf(test_base.TestBase.IsWindows(),
                 'https://github.com/bazelbuild/bazel/issues/5087')
class PyRemoteTest(test_base.TestBase):

  _worker_port = None

  def _RunRemoteBazel(self, args):
    return self.RunBazel(args + [
        '--spawn_strategy=remote',
        '--strategy=Javac=remote',
        '--strategy=Closure=remote',
        '--genrule_strategy=remote',
        '--define=EXECUTOR=remote',
        '--remote_executor=grpc://localhost:' + str(self._worker_port),
        '--remote_cache=grpc://localhost:' + str(self._worker_port),
        '--remote_timeout=3600',
        '--auth_enabled=false',
        '--remote_accept_cached=false',
    ])

  def setUp(self):
    test_base.TestBase.setUp(self)
    self._worker_port = self.StartRemoteWorker()

  def tearDown(self):
    self.StopRemoteWorker()
    test_base.TestBase.tearDown(self)

  def testPyTestRunsRemotely(self):
    self.ScratchFile('foo/BUILD', [
        'py_test(',
        '  name = "foo_test",',
        '  srcs = ["foo_test.py"],',
        ')',
    ])
    self.ScratchFile('foo/foo_test.py', [
        'print("Test ran")',
    ])

    # Test.
    _, stdout, _ = self._RunRemoteBazel(
        ['test', '--test_output=all', '//foo:foo_test']
    )
    self.assertIn('Test ran', stdout)

  # Regression test for https://github.com/bazelbuild/bazel/issues/9239
  def testPyTestWithStdlibCollisionRunsRemotely(self):
    self.ScratchFile('foo/BUILD', [
        'py_library(',
        '  name = "io",',
        '  srcs = ["io.py"],',
        ')',
        'py_test(',
        '  name = "io_test",',
        '  srcs = ["io_test.py"],',
        '  deps = [":io"],',
        ')',
    ])
    self.ScratchFile('foo/io.py', [
        'def my_func():',
        '  print("Test ran")',
    ])
    self.ScratchFile('foo/io_test.py', [
        'from foo import io',
        'io.my_func()',
    ])

    # Test.
    _, stdout, _ = self._RunRemoteBazel(
        ['test', '--test_output=all', '//foo:io_test']
    )
    self.assertIn('Test ran', stdout)


class PyRunfilesLibraryTest(test_base.TestBase):

  def testPyRunfilesLibraryCurrentRepository(self):
    self.ScratchFile('MODULE.bazel', [
        'local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")',
        'local_repository(name = "other_repo", path = "other_repo_path")',
    ])

    self.ScratchFile('pkg/BUILD.bazel', [
        'py_library(',
        '  name = "library",',
        '  srcs = ["library.py"],',
        '  visibility = ["//visibility:public"],',
        '  deps = ["@bazel_tools//tools/python/runfiles"],',
        ')',
        '',
        'py_binary(',
        '  name = "binary",',
        '  srcs = ["binary.py"],',
        '  deps = [',
        '    ":library",',
        '    "@bazel_tools//tools/python/runfiles",',
        '  ],',
        ')',
        '',
        'py_test(',
        '  name = "test",',
        '  srcs = ["test.py"],',
        '  deps = [',
        '    ":library",',
        '    "@bazel_tools//tools/python/runfiles",',
        '  ],',
        ')',
    ])
    self.ScratchFile('pkg/library.py', [
        'from bazel_tools.tools.python.runfiles import runfiles',
        'def print_repo_name():',
        '  print("in pkg/library.py: \'%s\'" % runfiles.Create().CurrentRepository())',
    ])
    self.ScratchFile('pkg/binary.py', [
        'from bazel_tools.tools.python.runfiles import runfiles',
        'from pkg import library',
        'library.print_repo_name()',
        'print("in pkg/binary.py: \'%s\'" % runfiles.Create().CurrentRepository())',
    ])
    self.ScratchFile('pkg/test.py', [
        'from bazel_tools.tools.python.runfiles import runfiles',
        'from pkg import library',
        'library.print_repo_name()',
        'print("in pkg/test.py: \'%s\'" % runfiles.Create().CurrentRepository())',
    ])

    self.ScratchFile('other_repo_path/REPO.bazel')
    self.ScratchFile('other_repo_path/pkg/BUILD.bazel', [
        'py_binary(',
        '  name = "binary",',
        '  srcs = ["binary.py"],',
        '  deps = [',
        '    "@//pkg:library",',
        '    "@bazel_tools//tools/python/runfiles",',
        '  ],',
        ')',
        '',
        'py_test(',
        '  name = "test",',
        '  srcs = ["test.py"],',
        '  deps = [',
        '    "@//pkg:library",',
        '    "@bazel_tools//tools/python/runfiles",',
        '  ],',
        ')',
    ])
    self.ScratchFile('other_repo_path/pkg/binary.py', [
        'from bazel_tools.tools.python.runfiles import runfiles',
        'from pkg import library',
        'library.print_repo_name()',
        'print("in external/other_repo/pkg/binary.py: \'%s\'" % runfiles.Create().CurrentRepository())',
    ])
    self.ScratchFile('other_repo_path/pkg/test.py', [
        'from bazel_tools.tools.python.runfiles import runfiles',
        'from pkg import library',
        'library.print_repo_name()',
        'print("in external/other_repo/pkg/test.py: \'%s\'" % runfiles.Create().CurrentRepository())',
    ])

    _, stdout, _ = self.RunBazel(['run', '//pkg:binary'])
    self.assertIn('in pkg/binary.py: \'\'', stdout)
    self.assertIn('in pkg/library.py: \'\'', stdout)

    _, stdout, _ = self.RunBazel(
        ['test', '//pkg:test', '--test_output=streamed']
    )
    self.assertIn('in pkg/test.py: \'\'', stdout)
    self.assertIn('in pkg/library.py: \'\'', stdout)

    _, stdout, _ = self.RunBazel(['run', '@other_repo//pkg:binary'])
    self.assertIn('in external/other_repo/pkg/binary.py: \'_main~_repo_rules~other_repo\'',
                  stdout)
    self.assertIn('in pkg/library.py: \'\'', stdout)

    _, stdout, _ = self.RunBazel(
        ['test', '@other_repo//pkg:test', '--test_output=streamed']
    )
    self.assertIn('in external/other_repo/pkg/test.py: \'_main~_repo_rules~other_repo\'', stdout)
    self.assertIn('in pkg/library.py: \'\'', stdout)


if __name__ == '__main__':
  absltest.main()
