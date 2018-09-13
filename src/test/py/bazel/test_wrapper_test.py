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


class TestWrapperTest(test_base.TestBase):

  def _CreateMockWorkspace(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'sh_test(',
        '    name = "passing_test.bat",',
        '    srcs = ["passing.bat"],',
        ')',
        'sh_test(',
        '    name = "failing_test.bat",',
        '    srcs = ["failing.bat"],',
        ')',
        'sh_test(',
        '    name = "printing_test.bat",',
        '    srcs = ["printing.bat"],',
        ')',
    ])
    self.ScratchFile('foo/passing.bat', ['@exit /B 0'], executable=True)
    self.ScratchFile('foo/failing.bat', ['@exit /B 1'], executable=True)
    self.ScratchFile('foo/printing.bat', [
        '@echo lorem ipsum',
        '@echo TEST_SRCDIR=%TEST_SRCDIR%',
        '@echo TEST_TMPDIR=%TEST_TMPDIR%',
        '@echo USER=%USER%',
        ], executable=True)

  def _AssertPassingTest(self, flag):
    exit_code, _, stderr = self.RunBazel([
        'test',
        '//foo:passing_test.bat',
        '-t-',
        flag,
    ])
    self.AssertExitCode(exit_code, 0, stderr)

  def _AssertFailingTest(self, flag):
    exit_code, _, stderr = self.RunBazel([
        'test',
        '//foo:failing_test.bat',
        '-t-',
        flag,
    ])
    self.AssertExitCode(exit_code, 3, stderr)

  def _AssertPrintingTest(self, flag):
    exit_code, stdout, stderr = self.RunBazel([
        'test',
        '//foo:printing_test.bat',
        '--test_output=streamed',
        '-t-',
        flag,
    ])
    self.AssertExitCode(exit_code, 0, stderr)
    lorem = False
    for line in stdout + stderr:
      if 'lorem ipsum' in line:
        lorem = True
      if 'TEST_SRCDIR=' in line:
        srcdir = line[len('TEST_SRCDIR='):]
      if 'TEST_TMPDIR=' in line:
        tmpdir = line[len('TEST_TMPDIR='):]
      if 'USER=' in line:
        user = line[len('USER='):]
    if not lorem:
      self.fail('FAIL: output:\n%s\n---' % '\n'.join(stderr + stdout))
    if not os.path.isdir(srcdir):
      self.fail('FAIL: output:\n%s\n---' % '\n'.join(stderr + stdout))
    if not os.path.isfile(srcdir + '/MANIFEST'):
      self.fail('FAIL: output:\n%s\n---' % '\n'.join(stderr + stdout))
    if not os.path.isabs(srcdir):
      self.fail('FAIL: output:\n%s\n---' % '\n'.join(stderr + stdout))
    if not os.path.isdir(tmpdir):
      self.fail('FAIL: output:\n%s\n---' % '\n'.join(stderr + stdout))
    if not os.path.isabs(tmpdir):
      self.fail('FAIL: output:\n%s\n---' % '\n'.join(stderr + stdout))
    if not user:
      self.fail('FAIL: output:\n%s\n---' % '\n'.join(stderr + stdout))

  def testTestExecutionWithTestSetupShAndWithTestWrapperExe(self):
    self._CreateMockWorkspace()
    flag = '--nowindows_native_test_wrapper'
    self._AssertPassingTest(flag)
    self._AssertFailingTest(flag)
    self._AssertPrintingTest(flag)
    # As of 2018-09-11, the Windows native test runner can run simple tests and
    # export a few envvars, though it does not completely set up the test's
    # environment yet.
    flag = '--windows_native_test_wrapper'
    self._AssertPassingTest(flag)
    self._AssertFailingTest(flag)
    self._AssertPrintingTest(flag)


if __name__ == '__main__':
  unittest.main()
