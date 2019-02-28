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


if __name__ == '__main__':
  unittest.main()
