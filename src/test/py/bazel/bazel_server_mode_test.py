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

import unittest

from src.test.py.bazel import test_base


class BazelServerModeTest(test_base.TestBase):

  def testBazelServerMode(self):
    self.ScratchFile('WORKSPACE')

    _, stdout, _ = self.RunBazel(['info', 'server_pid'])
    pid1 = stdout[0]
    _, stdout, _ = self.RunBazel(['info', 'server_pid'])
    pid2 = stdout[0]
    self.assertEqual(pid1, pid2)


if __name__ == '__main__':
  unittest.main()
