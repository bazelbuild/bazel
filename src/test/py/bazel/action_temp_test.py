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
import unittest
from src.test.py.bazel import test_base


class ActionTempTest(test_base.TestBase):
  """Test that Bazel sets a TMP/TEMP/TMPDIR envvar for actions."""

  def testStandaloneBuildAction(self):
    self.AssertTempEnvvarWithSpawnStrategy('standalone')

  if not test_base.TestBase.IsWindows():

    def testLinuxOrDarwinSandboxedBuildAction(self):
      self.AssertTempEnvvarWithSpawnStrategy('sandboxed')

    def testProcessWrappedSandboxedBuildAction(self):
      self.AssertTempEnvvarWithSpawnStrategy('processwrapper-sandbox')

  # Helper methods start here -------------------------------------------------

  def AssertTempEnvvarWithSpawnStrategy(self, strategy):
    self._CreateWorkspace()
    strategies = self._SpawnStrategies()
    self.assertIn(strategy, strategies)
    outputs = self._BuildRules(strategy)
    self.assertEqual(len(outputs), 2)
    self._AssertOutputFileContents(outputs['genrule'])
    self._AssertOutputFileContents(outputs['skylark'])

  def _CreateWorkspace(self, build_flags=None):
    if test_base.TestBase.IsWindows():
      toolname = 'foo.cmd'
      toolsrc = [
          '@SETLOCAL ENABLEEXTENSIONS',
          '@echo ON',
          'if [%TMP%] == [] goto fail',
          'if [%TEMP%] == [] goto fail',
          '',
          'echo foo1 > %TMP%\\foo1.txt',
          'echo foo2 > %TEMP%\\foo2.txt',
          'type %TMP%\\foo1.txt >> %1',
          'type %TEMP%\\foo2.txt >> %1',
          'echo bar >> %1',
          'set >> %1',
          'exit /B 0',
          '',
          ':fail',
          'echo inner build failed because TMP or TEMP is null > %1',
          'exit /B 1',
      ]
    else:
      toolname = 'foo.sh'
      toolsrc = [
          '#!/bin/bash',
          'set -eu',
          'if [ -n "${TMPDIR:-}" ]; then',
          '  sleep 1',
          '  echo foo > "$TMPDIR/foo.txt"',
          '  cat "$TMPDIR/foo.txt" > "$1"',
          '  echo bar >> "$1"',
          '  env | sort >> "$1"',
          'else',
          '  echo "inner build failed because TMPDIR is null" > "$1"',
          'fi',
      ]

    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/' + toolname, toolsrc, executable=True)
    self.ScratchFile('foo/foo.bzl', [
        'def _impl(ctx):',
        '  ctx.actions.run(',
        '      executable=ctx.executable.tool,',
        '      arguments=[ctx.outputs.out.path],',
        '      outputs=[ctx.outputs.out])',
        '  return [DefaultInfo(files=depset([ctx.outputs.out]))]',
        '',
        'foo = rule(',
        '    implementation=_impl,',
        '    attrs={"tool": attr.label(executable=True, cfg="host",',
        '                              allow_files=True, single_file=True)},',
        '    outputs={"out": "%{name}.txt"},',
        ')',
    ])

    self.ScratchFile('foo/BUILD', [
        'load("//foo:foo.bzl", "foo")',
        '',
        'genrule(',
        '    name = "genrule",',
        '    tools = ["%s"],' % toolname,
        '    outs = ["genrule.txt"],',
        '    cmd = "$(location %s) $@",' % toolname,
        ')',
        '',
        'foo(',
        '    name = "skylark",',
        '    tool = "%s",' % toolname,
        ')',
    ])

  def _SpawnStrategies(self):
    """Returns the list of supported --spawn_strategy values."""
    # TODO(b/37617303): make test UI-independent
    exit_code, _, stderr = self.RunBazel([
        'build', '--color=no', '--curses=no', '--spawn_strategy=foo',
        '--noexperimental_ui'
    ])
    self.AssertExitCode(exit_code, 2, stderr)
    pattern = re.compile(
        r'^ERROR:.*is an invalid value for.*Valid values are: (.*)\.$')
    for line in stderr:
      m = pattern.match(line)
      if m:
        return set(e.strip() for e in m.groups()[0].split(','))
    return []

  def _BuildRules(self, strategy):

    def _ReadFile(path):
      with open(path, 'rt') as f:
        return [l.strip() for l in f]

    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-genfiles'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_genfiles = stdout[0]

    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-bin'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    # TODO(b/37617303): make test UI-independent
    exit_code, _, stderr = self.RunBazel([
        'build', '--verbose_failures', '--noexperimental_ui',
        '--spawn_strategy=%s' % strategy, '//foo:genrule', '//foo:skylark'
    ])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertTrue(
        os.path.exists(os.path.join(bazel_genfiles, 'foo/genrule.txt')))
    self.assertTrue(os.path.exists(os.path.join(bazel_bin, 'foo/skylark.txt')))

    return {
        'genrule': _ReadFile(os.path.join(bazel_genfiles, 'foo/genrule.txt')),
        'skylark': _ReadFile(os.path.join(bazel_bin, 'foo/skylark.txt'))
    }

  def _AssertOutputFileContents(self, lines):
    if test_base.TestBase.IsWindows():
      self.assertGreater(len(lines), 5)
      self.assertEqual(lines[0], 'foo1')
      self.assertEqual(lines[1], 'foo2')
      self.assertEqual(lines[2], 'bar')
      self.assertEqual(len([l for l in lines if l.startswith('TMP')]), 1)
      self.assertEqual(len([l for l in lines if l.startswith('TEMP')]), 1)
    else:
      self.assertGreater(len(lines), 3)
      self.assertEqual(lines[0], 'foo')
      self.assertEqual(lines[1], 'bar')
      self.assertEqual(len([l for l in lines if l.startswith('TMPDIR')]), 1)


if __name__ == '__main__':
  unittest.main()
