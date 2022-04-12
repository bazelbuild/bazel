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
import re
import time
import unittest
from src.test.py.bazel import test_base


class BazelCleanTest(test_base.TestBase):

  def testBazelClean(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'genrule(',
        '  name = "x",',
        '  outs = ["x.out"],',
        '  cmd = "touch $@",',
        ')',
    ])

    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-genfiles'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_genfiles = stdout[0]

    exit_code, stdout, stderr = self.RunBazel(['info', 'output_base'])
    self.AssertExitCode(exit_code, 0, stderr)
    output_base = stdout[0]

    # Repeat 10 times to ensure flaky error like
    # https://github.com/bazelbuild/bazel/issues/5907 are caught.
    for _ in range(0, 10):
      exit_code, _, stderr = self.RunBazel(['build', '//foo:x'])
      self.AssertExitCode(exit_code, 0, stderr)
      self.assertTrue(os.path.exists(
          os.path.join(bazel_genfiles, 'foo/x.out')))

      exit_code, _, stderr = self.RunBazel(['clean'])
      self.AssertExitCode(exit_code, 0, stderr)
      self.assertFalse(os.path.exists(
          os.path.join(bazel_genfiles, 'foo/x.out')))
      self.assertTrue(os.path.exists(output_base))

      exit_code, _, stderr = self.RunBazel(['build', '//foo:x'])
      self.AssertExitCode(exit_code, 0, stderr)
      self.assertTrue(os.path.exists(os.path.join(bazel_genfiles, 'foo/x.out')))

      exit_code, _, stderr = self.RunBazel(['clean', '--expunge'])
      self.AssertExitCode(exit_code, 0, stderr)
      self.assertFalse(os.path.exists(
          os.path.join(bazel_genfiles, 'foo/x.out')))
      self.assertFalse(os.path.exists(output_base))

  @unittest.skipIf(not test_base.TestBase.IsLinux(),
                   'Async clean only supported on Linux')
  def testBazelAsyncClean(self):
    self.ScratchFile('WORKSPACE')
    exit_code, _, stderr = self.RunBazel(['clean', '--async'])
    self.AssertExitCode(exit_code, 0, stderr)
    matcher = self._findMatch(' moved to (.*) for deletion', stderr)
    self.assertTrue(matcher, stderr)
    first_temp = matcher.group(1)
    self.assertTrue(first_temp, stderr)
    # Now do it again (we need to build to recreate exec root).
    self.RunBazel(['build'])
    exit_code, _, stderr = self.RunBazel(['clean', '--async'])
    self.AssertExitCode(exit_code, 0, stderr)
    matcher = self._findMatch(' moved to (.*) for deletion', stderr)
    self.assertTrue(matcher, stderr)
    second_temp = matcher.group(1)
    self.assertTrue(second_temp, stderr)
    # Two directories should be different.
    self.assertNotEqual(second_temp, first_temp, stderr)

  @unittest.skipIf(not test_base.TestBase.IsLinux(),
                   'Async clean only supported on Linux')
  def testBazelAsyncCleanWithReadonlyDirectories(self):
    self.ScratchFile('WORKSPACE')
    exit_code, _, stderr = self.RunBazel(['build'])
    self.AssertExitCode(exit_code, 0, stderr)
    exit_code, stdout, stderr = self.RunBazel(['info', 'execution_root'])
    self.AssertExitCode(exit_code, 0, stderr)
    execroot = stdout[0]
    readonly_dir = os.path.join(execroot, 'readonly')
    os.mkdir(readonly_dir)
    open(os.path.join(readonly_dir, 'somefile'), 'wb').close()
    os.chmod(readonly_dir, 0o555)
    exit_code, _, stderr = self.RunBazel(['clean', '--async'])
    matcher = self._findMatch(' moved to (.*) for deletion', stderr)
    self.assertTrue(matcher, stderr)
    temp = matcher.group(1)
    for _ in range(50):
      if not os.path.isdir(temp):
        break
      time.sleep(.1)
    else:
      self.fail('temporary directory not removed: {!r}'.format(stderr))

  def _findMatch(self, pattern, items):
    r = re.compile(pattern)
    for line in items:
      matcher = r.search(line)
      if matcher:
        return matcher
    return None


if __name__ == '__main__':
  unittest.main()
