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
import unittest
from src.test.py.bazel import test_base


class WindowsRemoteTest(test_base.TestBase):

  def _RunRemoteBazel(self, args, port, env_remove=None, env_add=None):
    return self.RunBazel(
        args + [
            '--spawn_strategy=remote',
            '--strategy=Javac=remote',
            '--strategy=Closure=remote',
            '--genrule_strategy=remote',
            '--define=EXECUTOR=remote',
            '--remote_executor=localhost:' + str(port),
            '--remote_cache=localhost:' + str(port),
            '--experimental_strict_action_env=true',
            '--remote_timeout=3600',
            '--auth_enabled=false',
            '--remote_accept_cached=false',
        ],
        env_remove=env_remove,
        env_add=env_add)

  # Check that a binary built remotely is runnable locally. Among other things,
  # this means the runfiles manifest, which is not present remotely, must exist
  # locally.
  def testBinaryRunnableLocally(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'sh_binary(',
        '  name = "foo",',
        '  srcs = ["foo.sh"],',
        '  data = ["//bar:bar.txt"],',
        ')',
    ])
    self.ScratchFile(
        'foo/foo.sh', [
            '#!/bin/bash',
            'echo hello shell',
        ], executable=True)
    self.ScratchFile('bar/BUILD', ['exports_files(["bar.txt"])'])
    self.ScratchFile('bar/bar.txt', ['hello'])

    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-bin'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    port = self.StartRemoteWorker()

    try:
      # Build.
      exit_code, stdout, stderr = self._RunRemoteBazel(['build', '//foo:foo'],
                                                       port)
      print('\n'.join(stdout))
      self.AssertExitCode(exit_code, 0, stderr)

      # Run.
      foo_bin = os.path.join(bazel_bin, 'foo', 'foo.exe')
      self.assertTrue(os.path.exists(foo_bin))
      exit_code, stdout, stderr = self.RunProgram([foo_bin])
      self.AssertExitCode(exit_code, 0, stderr)
      self.assertEqual(stdout, ['hello shell'])
    finally:
      # Always stop the worker so we obtain logs in case an assertion failed
      # above.
      self.StopRemoteWorker()


if __name__ == '__main__':
  unittest.main()
