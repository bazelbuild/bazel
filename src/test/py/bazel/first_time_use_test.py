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


if __name__ == '__main__':
  unittest.main()
