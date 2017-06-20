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
        'genrule(', '  name = "x",', '  outs = ["x.out"],',
        '  cmd = "touch $@",', ')'
    ])

    exit_code, stdout, _ = self.RunBazel(['info', 'bazel-genfiles'])
    self.assertEqual(exit_code, 0)
    bazel_genfiles = stdout[0]

    exit_code, stdout, _ = self.RunBazel(['info', 'output_base'])
    self.assertEqual(exit_code, 0)
    output_base = stdout[0]

    exit_code, _, _ = self.RunBazel(['build', '//foo:x'])
    self.assertEqual(exit_code, 0)
    self.assertTrue(os.path.exists(os.path.join(bazel_genfiles, 'foo/x.out')))

    exit_code, _, _ = self.RunBazel(['clean'])
    self.assertEqual(exit_code, 0)
    self.assertFalse(os.path.exists(os.path.join(bazel_genfiles, 'foo/x.out')))
    self.assertTrue(os.path.exists(output_base))

    exit_code, _, _ = self.RunBazel(['build', '//foo:x'])
    self.assertEqual(exit_code, 0)
    self.assertTrue(os.path.exists(os.path.join(bazel_genfiles, 'foo/x.out')))

    exit_code, _, _ = self.RunBazel(['clean', '--expunge'])
    self.assertEqual(exit_code, 0)
    self.assertFalse(os.path.exists(os.path.join(bazel_genfiles, 'foo/x.out')))
    self.assertFalse(os.path.exists(output_base))


if __name__ == '__main__':
  unittest.main()
