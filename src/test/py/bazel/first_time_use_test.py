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

import unittest

from src.test.py.bazel import test_base


class FirstTimeUseTest(test_base.TestBase):

  def _FailWithOutput(self, output):
    self.fail('FAIL:\n | %s\n---' % '\n | '.join(output))

  def testNoPythonRequirement(self):
    """Regression test for https://github.com/bazelbuild/bazel/issues/6463."""
    self.ScratchFile('WORKSPACE')
    exit_code, stdout, stderr = self.RunBazel(['info', 'release'])
    self.AssertExitCode(exit_code, 0, stderr)
    for line in stdout + stderr:
      if 'python' in line and 'not found on PATH' in line:
        self._FailWithOutput(stdout + stderr)

  def testNoBashRequiredForSimpleBazelRun(self):
    """Regression test for https://github.com/bazelbuild/bazel/issues/8229."""
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'py_binary(',
        '    name = "x",'
        '    srcs = ["x.py"],',
        ')',
    ])
    self.ScratchFile('foo/x.py', [
        'from __future__ import print_function',
        'print("hello python")',
    ])

    if test_base.TestBase.IsWindows():
      exit_code, stdout, stderr = self.RunBazel([
          'run',
          '--shell_executable=',
          '--noincompatible_windows_bashless_run_command',
          '//foo:x',
      ])
      self.AssertNotExitCode(exit_code, 0, stderr)
      found_error = False
      for line in stdout + stderr:
        if 'ERROR' in line and 'needs a shell' in line:
          found_error = True
          break
      if not found_error:
        self._FailWithOutput(stdout + stderr)

      exit_code, stdout, stderr = self.RunBazel([
          'run',
          '--shell_executable=',
          '--incompatible_windows_bashless_run_command',
          '//foo:x',
      ])
      self.AssertExitCode(exit_code, 0, stderr)
      found_output = False
      for line in stdout + stderr:
        if 'ERROR' in line and 'needs a shell' in line:
          self._FailWithOutput(stdout + stderr)
        if 'hello python' in line:
          found_output = True
          break
      if not found_output:
        self._FailWithOutput(stdout + stderr)
    else:
      # The --incompatible_windows_bashless_run_command should be a no-op on
      # platforms other than Windows.
      for flag in [
          '--incompatible_windows_bashless_run_command',
          '--noincompatible_windows_bashless_run_command'
      ]:
        exit_code, stdout, stderr = self.RunBazel([
            'run',
            '--shell_executable=',
            flag,
            '//foo:x',
        ])
        self.AssertNotExitCode(exit_code, 0, stderr)
        found_error = False
        for line in stdout + stderr:
          if 'ERROR' in line and 'needs a shell' in line:
            found_error = True
            break
        if not found_error:
          self._FailWithOutput(['flag=' + flag] + stdout + stderr)


if __name__ == '__main__':
  unittest.main()
