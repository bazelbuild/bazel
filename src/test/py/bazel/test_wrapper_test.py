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
from src.test.py.bazel import test_base


class BazelCleanTest(test_base.TestBase):

  def testTestExecutionWithTestSetupShAndWithTestWrapperExe(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'py_test(',
        '    name = "x_test",',
        '    srcs = ["x_test.py"],',
        ')',
    ])
    self.ScratchFile('foo/x_test.py', [
        'from __future__ import print_function',
        'import unittest',
        '',
        'class XTest(unittest.TestCase):',
        '    def testFoo(self):',
        '        print("lorem ipsum")',
        '',
        'if __name__ == "__main__":',
        '  unittest.main()',
    ], executable = True)

    # Run test with test-setup.sh
    exit_code, stdout, stderr = self.RunBazel([
        'test', '//foo:x_test', '--test_output=streamed', '-t-',
        '--nowindows_native_test_wrapper'])
    self.AssertExitCode(exit_code, 0, stderr)
    found = False
    for line in stdout + stderr:
        if 'lorem ipsum' in line:
            found = True
    if not found:
        self.fail('FAIL: output:\n%s\n---' % '\n'.join(stderr + stdout))

    # Run test with test_wrapper.exe
    exit_code, _, stderr = self.RunBazel([
        'test', '//foo:x_test', '--test_output=streamed', '-t-',
        '--windows_native_test_wrapper'])

    # As of 2018-08-07, test_wrapper.exe cannot yet run tests.
    self.AssertExitCode(exit_code, 3, stderr)

if __name__ == '__main__':
  unittest.main()

