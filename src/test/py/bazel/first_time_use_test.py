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

from absl.testing import absltest
from src.test.py.bazel import test_base


class FirstTimeUseTest(test_base.TestBase):

  def _FailWithOutput(self, output):
    self.fail('FAIL:\n | %s\n---' % '\n | '.join(output))

  def testNoPythonRequirement(self):
    """Regression test for https://github.com/bazelbuild/bazel/issues/6463."""
    _, stdout, stderr = self.RunBazel(['info', 'release'])
    for line in stdout + stderr:
      if 'python' in line and 'not found on PATH' in line:
        self._FailWithOutput(stdout + stderr)

  def _AssertBazelRunBinaryOutput(self, exit_code, stdout, stderr):
    self.AssertExitCode(exit_code, 0, stderr)
    found_hello = found_arg_a = found_arg_bc = False
    for line in stdout + stderr:
      if 'ERROR' in line and 'needs a shell' in line:
        self._FailWithOutput(stdout + stderr)
      if not found_hello and 'hello python' in line:
        found_hello = True
      elif not found_arg_a and 'arg[1]=(a)':
        found_arg_a = True
      elif not found_arg_bc and 'arg[2]=(b c)':
        found_arg_bc = True
        break
    if not found_hello or not found_arg_a or not found_arg_bc:
      self._FailWithOutput(stdout + stderr)

  def testNoBashRequiredForSimpleBazelRun(self):
    """Regression test for https://github.com/bazelbuild/bazel/issues/8229."""
    self.ScratchFile('foo/BUILD', [
        'py_binary(',
        '    name = "x",'
        '    srcs = ["x.py"],',
        '    args = ["a", "\'b c\'"],',
        ')',
    ])
    self.ScratchFile('foo/x.py', [
        'from __future__ import print_function',
        'import sys',
        'print("hello python")',
        'for i in range(len(sys.argv)):',
        '    print("arg%d=(%s)" % (i, sys.argv[i]))',
    ])

    if test_base.TestBase.IsWindows():
      # "bazel run" needs no Bash on Windows, so this call should succeed.
      exit_code, stdout, stderr = self.RunBazel(
          [
              'run',
              '--shell_executable=',
              '//foo:x',
          ],
          allow_failure=True,
      )
      self._AssertBazelRunBinaryOutput(exit_code, stdout, stderr)
    else:
      exit_code, stdout, stderr = self.RunBazel(
          [
              'run',
              # Run fails because we provide no shell. Platforms other than
              # Windows always use Bash for "bazel run".
              '--shell_executable=',
              '//foo:x',
          ],
          allow_failure=True,
      )
      self.AssertNotExitCode(exit_code, 0, stderr)
      found_error = False
      for line in stdout + stderr:
        if 'ERROR' in line and 'needs a shell' in line:
          found_error = True
          break
      if not found_error:
        self._FailWithOutput(stdout + stderr)

      # Run succeeds because there is a shell.
      exit_code, stdout, stderr = self.RunBazel(
          ['run', '//foo:x'], allow_failure=True
      )
      self._AssertBazelRunBinaryOutput(exit_code, stdout, stderr)


if __name__ == '__main__':
  absltest.main()
