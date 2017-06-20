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


class BazelWindowsTest(test_base.TestBase):

  def testWindowsUnixRoot(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', ['cc_binary(name="x", srcs=["x.cc"])'])
    self.ScratchFile('foo/x.cc', [
        '#include <stdio.h>', 'int main(int, char**) {'
        '  printf("hello\\n");', '  return 0;', '}'
    ])

    exit_code, _, stderr = self.RunBazel(
        ['--batch', 'build', '//foo:x', '--cpu=x64_windows_msys'],
        env_remove={'BAZEL_SH'})
    self.assertEqual(exit_code, 2)
    self.assertIn('\'BAZEL_SH\' environment variable is not set',
                  '\n'.join(stderr))

    exit_code, _, stderr = self.RunBazel([
        '--batch', '--host_jvm_args=-Dbazel.windows_unix_root=', 'build',
        '//foo:x', '--cpu=x64_windows_msys'
    ])
    self.assertEqual(exit_code, 37)
    self.assertIn('"bazel.windows_unix_root" JVM flag is not set',
                  '\n'.join(stderr))

    exit_code, _, _ = self.RunBazel(
        ['--batch', 'build', '//foo:x', '--cpu=x64_windows_msys'])
    self.assertEqual(exit_code, 0)


if __name__ == '__main__':
  unittest.main()
