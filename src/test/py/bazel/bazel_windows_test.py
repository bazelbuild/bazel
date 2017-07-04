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


class BazelWindowsTest(test_base.TestBase):

  def createProjectFiles(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', ['cc_binary(name="x", srcs=["x.cc"])'])
    self.ScratchFile('foo/x.cc', [
        '#include <stdio.h>',
        'int main(int, char**) {'
        '  printf("hello\\n");',
        '  return 0;',
        '}',
    ])

  def testWindowsUnixRoot(self):
    self.createProjectFiles()
    exit_code, _, stderr = self.RunBazel(
        ['--batch', 'build', '//foo:x', '--cpu=x64_windows_msys'],
        env_remove={'BAZEL_SH'})
    self.AssertExitCode(exit_code, 2, stderr)
    self.assertIn('\'BAZEL_SH\' environment variable is not set',
                  '\n'.join(stderr))

    exit_code, _, stderr = self.RunBazel([
        '--batch', '--host_jvm_args=-Dbazel.windows_unix_root=', 'build',
        '//foo:x', '--cpu=x64_windows_msys'
    ])
    self.AssertExitCode(exit_code, 37, stderr)
    self.assertIn('"bazel.windows_unix_root" JVM flag is not set',
                  '\n'.join(stderr))

    exit_code, _, stderr = self.RunBazel(
        ['--batch', 'build', '//foo:x', '--cpu=x64_windows_msys'])
    self.AssertExitCode(exit_code, 0, stderr)

  def testUseMSVCWrapperScript(self):
    self.createProjectFiles()

    exit_code, stdout, stderr = self.RunBazel(['info', 'execution_root'])
    self.AssertExitCode(exit_code, 0, stderr)
    execution_root = stdout[0]

    exit_code, _, stderr = self.RunBazel(
        [
            '--batch',
            'build',
            '//foo:x',
        ],
        # USE_MSVC_WRAPPER will be needed after
        # swichting wrapper-less CROSSTOOL as default
        env_add={'USE_MSVC_WRAPPER': '1'},)
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertTrue(
        os.path.exists(
            os.path.join(
                execution_root,
                'external/local_config_cc/wrapper/bin/pydir/msvc_tools.py')))


if __name__ == '__main__':
  unittest.main()
