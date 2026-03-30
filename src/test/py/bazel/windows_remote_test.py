# pylint: disable=g-bad-file-header
# pylint: disable=superfluous-parens
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
from absl.testing import absltest
from src.test.py.bazel import test_base


class WindowsRemoteTest(test_base.TestBase):

  _worker_port = None

  def _RunRemoteBazel(self, args, env_remove=None, env_add=None):
    return self.RunBazel(
        args + [
            '--spawn_strategy=remote',
            '--strategy=Javac=remote',
            '--strategy=Closure=remote',
            '--genrule_strategy=remote',
            '--remote_executor=grpc://localhost:' + str(self._worker_port),
            '--remote_cache=grpc://localhost:' + str(self._worker_port),
            '--remote_timeout=3600',
            '--auth_enabled=false',
            '--remote_accept_cached=false',
        ],
        env_remove=env_remove,
        env_add=env_add)

  def setUp(self):
    test_base.TestBase.setUp(self)
    self.ScratchFile(
        'MODULE.bazel', ["bazel_dep(name = 'rules_shell', version = '0.1.1')"]
    )
    self._worker_port = self.StartRemoteWorker()

  def tearDown(self):
    test_base.TestBase.tearDown(self)
    self.StopRemoteWorker()

  # Check that a binary built remotely is runnable locally. Among other things,
  # this means the runfiles manifest, which is not present remotely, must exist
  # locally.
  def testBinaryRunsLocally(self):
    self.ScratchFile(
        'foo/BUILD',
        [
            'load("@rules_shell//shell:sh_binary.bzl", "sh_binary")',
            'sh_binary(',
            '  name = "foo",',
            '  srcs = ["foo.sh"],',
            '  data = ["//bar:bar.txt"],',
            ')',
        ],
    )
    self.ScratchFile(
        'foo/foo.sh', [
            '#!/bin/sh',
            'echo hello shell',
        ], executable=True)
    self.ScratchFile('bar/BUILD', ['exports_files(["bar.txt"])'])
    self.ScratchFile('bar/bar.txt', ['hello'])

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]

    # Build.
    self._RunRemoteBazel(['build', '//foo:foo'])

    # Run.
    foo_bin = os.path.join(bazel_bin, 'foo', 'foo.exe')
    self.assertTrue(os.path.exists(foo_bin))
    _, stdout, _ = self.RunProgram([foo_bin])
    self.assertEqual(stdout, ['hello shell'])

  def testShTestRunsLocally(self):
    self.ScratchFile(
        'foo/BUILD',
        [
            'load("@rules_shell//shell:sh_test.bzl", "sh_test")',
            'sh_test(',
            '  name = "foo_test",',
            '  srcs = ["foo_test.sh"],',
            '  data = ["//bar:bar.txt"],',
            ')',
        ],
    )
    self.ScratchFile(
        'foo/foo_test.sh', ['#!/bin/sh', 'echo hello test'], executable=True)
    self.ScratchFile('bar/BUILD', ['exports_files(["bar.txt"])'])
    self.ScratchFile('bar/bar.txt', ['hello'])

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]

    # Build.
    self._RunRemoteBazel(['build', '--test_output=all', '//foo:foo_test'])

    # Test.
    foo_test_bin = os.path.join(bazel_bin, 'foo', 'foo_test.exe')
    self.assertTrue(os.path.exists(foo_test_bin))
    self.RunProgram([foo_test_bin])

  # Remotely, the runfiles manifest does not exist.
  def testShTestRunsRemotely(self):
    self.ScratchFile(
        'foo/BUILD',
        [
            'load("@rules_shell//shell:sh_test.bzl", "sh_test")',
            'sh_test(',
            '  name = "foo_test",',
            '  srcs = ["foo_test.sh"],',
            '  data = ["//bar:bar.txt"],',
            ')',
        ],
    )
    self.ScratchFile(
        'foo/foo_test.sh', [
            '#!/bin/sh',
            'echo hello test',
            'echo "RUNFILES_MANIFEST_FILE: \\"${RUNFILES_MANIFEST_FILE:-}\\""'
        ],
        executable=True)
    self.ScratchFile('bar/BUILD', ['exports_files(["bar.txt"])'])
    self.ScratchFile('bar/bar.txt', ['hello'])

    # Test.
    _, stdout, _ = self._RunRemoteBazel(
        ['test', '--test_output=all', '//foo:foo_test']
    )
    self.assertIn('RUNFILES_MANIFEST_FILE: ""', stdout)

  # The Java launcher uses Rlocation which has differing behavior for local and
  # remote.
  def testJavaTestRunsRemotely(self):
    self.ScratchFile('foo/BUILD', [
        'java_test(',
        '  name = "foo_test",',
        '  srcs = ["TestFoo.java"],',
        '  main_class = "TestFoo",',
        '  use_testrunner = 0,',
        '  data = ["//bar:bar.txt"],',
        ')',
    ])
    self.ScratchFile(
        'foo/TestFoo.java', [
            'public class TestFoo {',
            'public static void main(String[] args) {',
            'System.out.println("hello java test");',
            '}',
            '}',
        ],
        executable=True)
    self.ScratchFile('bar/BUILD', ['exports_files(["bar.txt"])'])
    self.ScratchFile('bar/bar.txt', ['hello'])

    # Test.
    self._RunRemoteBazel(['test', '--test_output=all', '//foo:foo_test'])

  # Exercises absolute path handling in Rlocation.
  # This depends on there being a Java installation to c:\openjdk. If you have
  # it elsewhere, add --test_env=JAVA_HOME to your Bazel invocation to fix this
  # test.
  def testJavaTestWithRuntimeRunsRemotely(self):
    self.ScratchFile('foo/BUILD', [
        'package(default_visibility = ["//visibility:public"])',
        'java_test(',
        '  name = "foo_test",',
        '  srcs = ["TestFoo.java"],',
        '  main_class = "TestFoo",',
        '  use_testrunner = 0,',
        '  data = ["//bar:bar.txt"],',
        ')',
    ])
    self.ScratchFile(
        'foo/TestFoo.java', [
            'public class TestFoo {',
            'public static void main(String[] args) {',
            'System.out.println("hello java test");',
            '}',
            '}',
        ],
        executable=True)
    self.ScratchFile('bar/BUILD', ['exports_files(["bar.txt"])'])
    self.ScratchFile('bar/bar.txt', ['hello'])

    # Test.
    self._RunRemoteBazel(['test', '--test_output=all', '//foo:foo_test'])

  # Genrules are notably different than tests because RUNFILES_DIR is not set
  # for genrule tool launchers, so the runfiles directory is discovered based on
  # the executable path.
  def testGenruleWithToolRunsRemotely(self):
    # TODO(jsharpe): Replace sh_binary with py_binary once
    # https://github.com/bazelbuild/bazel/issues/5087 resolved.
    self.ScratchFile(
        'foo/BUILD',
        [
            'load("@rules_shell//shell:sh_binary.bzl", "sh_binary")',
            'sh_binary(',
            '  name = "data_tool",',
            '  srcs = ["data_tool.sh"],',
            '  data = ["//bar:bar.txt"],',
            ')',
            'sh_binary(',
            '  name = "tool",',
            '  srcs = ["tool.sh"],',
            '  data = [":data_tool"],',
            ')',
            'genrule(',
            '  name = "genrule",',
            '  srcs = [],',
            '  outs = ["out.txt"],',
            '  cmd  = "$(location :tool) > \\"$@\\"",',
            '  tools = [":tool"],',
            ')',
        ],
    )
    self.ScratchFile(
        'foo/tool.sh', [
            '#!/bin/sh',
            'echo hello tool',
            # TODO(jsharpe): This is kind of an ugly way to call the data
            # dependency, but the best I can find. Instead, use py_binary +
            # Python runfiles library here once that's possible.
            '$RUNFILES_DIR/_main/foo/data_tool',
        ],
        executable=True)
    self.ScratchFile(
        'foo/data_tool.sh', [
            '#!/bin/sh',
            'echo hello data tool',
        ],
        executable=True)
    self.ScratchFile('bar/BUILD', ['exports_files(["bar.txt"])'])
    self.ScratchFile('bar/bar.txt', ['hello'])

    # Build.
    self._RunRemoteBazel(['build', '//foo:genrule'])

  # TODO(jsharpe): Add a py_test example here. Blocked on
  # https://github.com/bazelbuild/bazel/issues/5087


if __name__ == '__main__':
  absltest.main()
