# Copyright 2019 The Bazel Authors. All rights reserved.
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


class GenRuleTest(test_base.TestBase):

  def testCopyWithBashAndBatch(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'genrule(',
        '  name = "x",',
        '  srcs = ["hello"],',
        '  outs = ["hello_copied"],',
        '  cmd_bash = "cp $< $@",',
        '  cmd_bat = "copy $< $@",',
        ')',
    ])
    self.ScratchFile('foo/hello', ['hello world'])

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]

    self.RunBazel(['build', '//foo:x'])

    copied = os.path.join(bazel_bin, 'foo', 'hello_copied')
    self.assertTrue(os.path.exists(copied))
    self.AssertFileContentContains(copied, 'hello world')

  def testCopyWithBashAndPowershell(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'genrule(',
        '  name = "x",',
        '  srcs = ["hello"],',
        '  outs = ["hello_copied"],',
        '  cmd_bash = "cp $< $@",',
        '  cmd_ps = "Copy-Item $< -Destination $@",',
        ')',
    ])
    self.ScratchFile('foo/hello', ['hello world'])

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]

    self.RunBazel(['build', '//foo:x'])

    copied = os.path.join(bazel_bin, 'foo', 'hello_copied')
    self.assertTrue(os.path.exists(copied))
    self.AssertFileContentContains(copied, 'hello world')

  def testShOptionOverridesDefault(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'genrule(',
        '  name = "x",',
        '  outs = ["hello"],',
        '  cmd = "echo hello > $@"',
        ')',
    ])
    # Build this target and make sure it passes with the default sh config
    self.RunBazel(['build', '//foo:x'])
    # Pass a bad --sh_executable and ensure this causes the build to fail
    exit_code, _, stderr = self.RunBazel(
        ['build', '//foo:x', '--shell_executable=fake_executable_should_fail'],
        allow_failure=True,
    )
    self.assertNotEqual(exit_code, 0)
    self.assertIn('fake_executable_should_fail', ''.join(stderr))

  def testScriptFileIsUsedWithBatch(self):
    if not self.IsWindows():
      return
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'genrule(',
        '  name = "x",',
        '  outs = ["hello_world"],',
        '  cmd_bat = \'&& \'.join([\"echo Hello world>>$(location hello_world)\" for _ in range(0, 1000)]),',
        ')',
    ])

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]

    self.RunBazel(['build', '//foo:x'])

    script = os.path.join(bazel_bin, 'foo', 'x.genrule_script.bat')
    hello = os.path.join(bazel_bin, 'foo', 'hello_world')
    self.assertTrue(os.path.exists(script))
    self.assertTrue(os.path.exists(hello))

    expected_content = '\n'.join(['Hello world' for _ in range(0, 1000)])
    self.AssertFileContentContains(hello, expected_content)

  def testScriptFileIsUsedWithPowershell(self):
    if not self.IsWindows():
      return
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'genrule(',
        '  name = "x",',
        '  outs = ["hello_world"],',
        '  cmd_ps ="; ".join(["echo \\"Hello world\\">>$(location hello_world)" for _ in range(0, 1000)]),',
        ')',
    ])

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]

    self.RunBazel(['build', '//foo:x'])

    script = os.path.join(bazel_bin, 'foo', 'x.genrule_script.ps1')
    hello = os.path.join(bazel_bin, 'foo', 'hello_world')
    self.assertTrue(os.path.exists(script))
    self.assertTrue(os.path.exists(hello))

    expected_content = '\n'.join(['Hello world' for _ in range(0, 1000)])
    self.AssertFileContentContains(hello, expected_content)

  def testCommandFailsEagerlyInPowershell(self):
    if not self.IsWindows():
      return
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'genrule(',
        '  name = "x",',
        '  outs = ["hello_world"],',
        '  cmd_ps = "echo hello >$@; command_not_exist; echo world >>$@;",',
        ')',
    ])

    exit_code, _, stderr = self.RunBazel(
        ['build', '//foo:x'], allow_failure=True
    )
    self.AssertExitCode(exit_code, 1, stderr)
    self.assertIn(
        'The term \'command_not_exist\' is not recognized as the name of a cmdlet',
        ''.join(stderr))

  def testCopyWithSpacesWithBatch(self):
    if not self.IsWindows():
      return
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'genrule(',
        '  name = "x",',
        '  srcs = ["hello source"],',
        '  outs = ["hello copied"],',
        '  cmd_bat = "copy \\"$<\\" \\"$@\\"",',
        ')',
    ])
    self.ScratchFile('foo/hello source', ['hello world'])

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]

    self.RunBazel(['build', '//foo:x'])

    copied = os.path.join(bazel_bin, 'foo', 'hello copied')
    self.assertTrue(os.path.exists(copied))
    self.AssertFileContentContains(copied, 'hello world')

  def testCopyWithSpacesWithPowershell(self):
    if not self.IsWindows():
      return
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'genrule(',
        '  name = "x",',
        '  srcs = ["hello source"],',
        '  outs = ["hello copied"],',
        '  cmd_ps = "cp \\"$<\\" \\"$@\\"",',
        ')',
    ])
    self.ScratchFile('foo/hello source', ['hello world'])

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]

    self.RunBazel(['build', '//foo:x'])

    copied = os.path.join(bazel_bin, 'foo', 'hello copied')
    self.assertTrue(os.path.exists(copied))
    self.AssertFileContentContains(copied, 'hello world')


if __name__ == '__main__':
  unittest.main()
