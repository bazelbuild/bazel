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
import unittest
import zipfile
from src.test.py.bazel import test_base


class PyTest(test_base.TestBase):
  """Integration tests for the Python rules of Bazel."""

  def createSimpleFiles(self):
    self.CreateWorkspaceWithDefaultRepos('WORKSPACE')

    self.ScratchFile(
        'a/BUILD',
        [
            'py_binary(name="a", srcs=["a.py"], deps=[":b"])',
            'py_library(name="b", srcs=["b.py"])',
        ])

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
    exit_code, stdout, stderr = self.RunBazel(['run', '//a:a'])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertTrue('Hello, World' in stdout)

  def testRunfilesSymlinks(self):
    if test_base.TestBase.IsWindows():
      # No runfiles symlinks on Windows
      return

    self.createSimpleFiles()
    exit_code, _, stderr = self.RunBazel(['build', '//a:a'])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertTrue(os.path.isdir('bazel-bin/a/a.runfiles'))
    self.assertTrue(os.readlink('bazel-bin/a/a.runfiles/__main__/a/a.py')
                    .endswith('/a/a.py'))
    self.assertTrue(os.readlink('bazel-bin/a/a.runfiles/__main__/a/b.py')
                    .endswith('/a/b.py'))


class TestInitPyFiles(test_base.TestBase):

  def createSimpleFiles(self, create_init=True):
    self.CreateWorkspaceWithDefaultRepos('WORKSPACE')

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
    exit_code, _, stderr = self.RunBazel(['build', '//src/a:a'])
    self.AssertExitCode(exit_code, 0, stderr)
    if self.IsWindows():
      # On Windows Bazel creates bazel-bin/src/a/a.zip
      self.assertTrue(os.path.exists('bazel-bin/src/a/a.zip'))
      with zipfile.ZipFile('bazel-bin/src/a/a.zip', 'r') as z:
        zip_contents = set(z.namelist())
      self.assertIn('runfiles/__main__/src/__init__.py', zip_contents)
      self.assertIn('runfiles/__main__/src/a/__init__.py', zip_contents)
    else:
      self.assertTrue(
          os.path.exists('bazel-bin/src/a/a.runfiles/__main__/src/__init__.py'))
      self.assertTrue(
          os.path.exists(
              'bazel-bin/src/a/a.runfiles/__main__/src/a/__init__.py'))

  def testInitPyFilesNotCreatedWhenLegacyCreateInitIsSet(self):
    self.createSimpleFiles(create_init=False)
    exit_code, _, stderr = self.RunBazel(['build', '//src/a:a'])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertFalse(
        os.path.exists('bazel-bin/src/a/a.runfiles/__main__/src/__init__.py'))
    self.assertFalse(
        os.path.exists('bazel-bin/src/a/a.runfiles/__main__/src/a/__init__.py'))

  # Regression test for https://github.com/bazelbuild/bazel/pull/10119
  def testBuildingZipFileWithTargetNameWithDot(self):
    self.CreateWorkspaceWithDefaultRepos('WORKSPACE')
    self.ScratchFile('BUILD', [
        'py_binary(',
        '  name = "bin.v1",  # .v1 should not be treated as extension and removed accidentally',
        '  srcs = ["bin.py"],',
        '  main = "bin.py",',
        ')',
    ])
    self.ScratchFile('bin.py', 'print("Hello, world")')
    exit_code, _, stderr = self.RunBazel(
        ['build', '--build_python_zip', '//:bin.v1'])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertTrue(os.path.exists('bazel-bin/bin.v1.temp'))
    self.assertTrue(os.path.exists('bazel-bin/bin.v1.zip'))


@unittest.skipIf(test_base.TestBase.IsWindows(),
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
    self.CreateWorkspaceWithDefaultRepos('WORKSPACE')
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
    exit_code, stdout, stderr = self._RunRemoteBazel(
        ['test', '--test_output=all', '//foo:foo_test'])
    self.AssertExitCode(exit_code, 0, stderr, stdout)
    self.assertIn('Test ran', stdout)

  # Regression test for https://github.com/bazelbuild/bazel/issues/9239
  def testPyTestWithStdlibCollisionRunsRemotely(self):
    self.CreateWorkspaceWithDefaultRepos('WORKSPACE')
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
    exit_code, stdout, stderr = self._RunRemoteBazel(
        ['test', '--test_output=all', '//foo:io_test'])
    self.AssertExitCode(exit_code, 0, stderr, stdout)
    self.assertIn('Test ran', stdout)


if __name__ == '__main__':
  unittest.main()
