# Copyright 2026 The Bazel Authors. All rights reserved.
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
"""Tests the `stdout` parameter of ctx.actions.run under local execution."""

import os
from absl.testing import absltest
from src.test.py.bazel import test_base


class ActionStdoutTest(test_base.TestBase):
  """Tests that ctx.actions.run captures stdout under unsandboxed local execution."""

  def _CreateWorkspace(self):
    if test_base.TestBase.IsWindows():
      toolname = 'tool.cmd'
      toolsrc = [
          '@echo off',
          'echo hello-from-stdout',
          'echo hello-from-stderr 1>&2',
          'if "%1"=="fail" exit /B 1',
          'exit /B 0',
      ]
    else:
      toolname = 'tool.sh'
      toolsrc = [
          '#!/bin/bash',
          'echo hello-from-stdout',
          'echo hello-from-stderr 1>&2',
          'if [ "$1" = "fail" ]; then exit 1; fi',
      ]

    self.ScratchFile('MODULE.bazel')
    self.ScratchFile('foo/' + toolname, toolsrc, executable=True)
    self.ScratchFile(
        'foo/defs.bzl',
        [
            'def _impl(ctx):',
            '    out = ctx.actions.declare_file(ctx.attr.name + ".out")',
            '    ctx.actions.run(',
            '        outputs = [],',
            '        executable = ctx.executable.tool,',
            '        arguments = [ctx.attr.mode],',
            '        stdout = out,',
            '        mnemonic = "Capture",',
            '    )',
            '    return [DefaultInfo(files = depset([out]))]',
            '',
            'capture = rule(',
            '    implementation = _impl,',
            '    attrs = {',
            '        "mode": attr.string(default = "ok"),',
            '        "tool": attr.label(',
            '            allow_single_file = True,',
            '            executable = True,',
            '            cfg = "exec",',
            '        ),',
            '    },',
            ')',
        ],
    )
    self.ScratchFile(
        'foo/BUILD',
        [
            'load(":defs.bzl", "capture")',
            '',
            'capture(',
            '    name = "captured",',
            '    tool = "%s",' % toolname,
            ')',
            '',
            'capture(',
            '    name = "captured_fail",',
            '    mode = "fail",',
            '    tool = "%s",' % toolname,
            ')',
        ],
    )

  def testStdoutCaptured(self):
    self._CreateWorkspace()
    _, stdout, stderr = self.RunBazel(
        ['build', '--spawn_strategy=standalone', '//foo:captured']
    )
    output = '\n'.join(stdout + stderr)
    # The tool's stderr is reported as regular action output, but its stdout
    # is captured into the output file instead.
    self.assertIn('hello-from-stderr', output)
    self.assertNotIn('hello-from-stdout', output)

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]
    self.AssertFileContentContains(
        os.path.join(bazel_bin, 'foo', 'captured.out'), 'hello-from-stdout'
    )

  def testFailingActionDoesNotReportCapturedStdout(self):
    self._CreateWorkspace()
    exit_code, stdout, stderr = self.RunBazel(
        ['build', '--spawn_strategy=standalone', '//foo:captured_fail'],
        allow_failure=True,
    )
    self.AssertNotExitCode(exit_code, 0, stderr)
    output = '\n'.join(stdout + stderr)
    # The failing action's stderr is reported as usual, but its captured
    # stdout is not.
    self.assertIn('hello-from-stderr', output)
    self.assertNotIn('hello-from-stdout', output)


if __name__ == '__main__':
  absltest.main()
