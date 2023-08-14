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

  _invalidations = 0

  def testActionTemp(self):
    self._CreateWorkspace()
    strategies = self._SpawnStrategies()

    self.assertIn('standalone', strategies)
    if not test_base.TestBase.IsWindows():
      self.assertIn('sandboxed', strategies)
      self.assertIn('processwrapper-sandbox', strategies)

    bazel_bin = self._BazelOutputDirectory('bazel-bin')
    bazel_genfiles = self._BazelOutputDirectory('bazel-genfiles')

    # Test without user-defined temp directory.
    # In the absence of TMP/TEMP/TMPDIR, the LocalEnvProvider implementations
    # set the fallback temp directory.
    if test_base.TestBase.IsWindows():
      expected_tmpdir_regex = r'execroot\\.+\\local-spawn-runner.[0-9]+\\work$'
    else:
      expected_tmpdir_regex = '^/tmp$'

    self._AssertTempDir('standalone', expected_tmpdir_regex, bazel_bin,
                        bazel_genfiles)
    if not test_base.TestBase.IsWindows():
      self._AssertTempDir('sandboxed', expected_tmpdir_regex, bazel_bin,
                          bazel_genfiles)
      self._AssertTempDir('processwrapper-sandbox', expected_tmpdir_regex,
                          bazel_bin, bazel_genfiles)

    # Test with user-defined temp directory.
    self._AssertClientEnvTemp('standalone', bazel_bin, bazel_genfiles)
    if not test_base.TestBase.IsWindows():
      self._AssertClientEnvTemp('sandboxed', bazel_bin, bazel_genfiles)
      self._AssertClientEnvTemp('processwrapper-sandbox', bazel_bin,
                                bazel_genfiles)

  # Helper methods start here -------------------------------------------------

  def _AssertClientEnvTemp(self, strategy, bazel_bin, bazel_genfiles):

    def _Impl(tmp_dir):
      self._AssertTempDir(
          strategy=strategy,
          expected_tmpdir_regex=os.path.basename(tmp_dir),
          bazel_bin=bazel_bin,
          bazel_genfiles=bazel_genfiles,
          env_add=dict((k, tmp_dir) for k in self._TempEnvvars()))

    _Impl(self.ScratchDir(strategy + '-temp-1'))
    # Assert that the actions pick up the current client environment.
    # Check this by invalidating the actions (update input.txt) and running
    # Bazel with a different environment.
    _Impl(self.ScratchDir(strategy + '-temp-2'))

  def _AssertTempDir(self,
                     strategy,
                     expected_tmpdir_regex,
                     bazel_bin,
                     bazel_genfiles,
                     env_add=None):
    self._invalidations += 1
    input_file_contents = str(self._invalidations)
    self._UpdateInputFile(input_file_contents)
    outputs = self._BuildRules(
        strategy,
        bazel_bin,
        bazel_genfiles,
        env_remove=self._TempEnvvars(),
        env_add=env_add)
    self.assertEqual(len(outputs), 2)
    self._AssertOutputFileContents(outputs['genrule'], input_file_contents,
                                   expected_tmpdir_regex)
    self._AssertOutputFileContents(outputs['starlark'], input_file_contents,
                                   expected_tmpdir_regex)

  def _UpdateInputFile(self, content):
    self.ScratchFile('foo/input.txt', [content])

  def _TempEnvvars(self):
    if test_base.TestBase.IsWindows():
      return ['TMP', 'TEMP']
    else:
      return ['TMPDIR']

  def _BazelOutputDirectory(self, info_key):
    _, stdout, _ = self.RunBazel(['info', info_key])
    return stdout[0]

  def _InvalidateActions(self, content):
    self.ScratchFile('foo/input.txt', [content])

  def _CreateWorkspace(self, build_flags=None):
    if test_base.TestBase.IsWindows():
      toolname = 'foo.cmd'
      toolsrc = [
          '@SETLOCAL ENABLEEXTENSIONS',
          '@echo ON',
          'if [%TMP%] == [] exit /B 1',
          'if [%TEMP%] == [] exit /B 1',
          'if not exist "%2" exit /B 2',
          'set input_file=%2',
          # TMP/TEMP may refer to directories that other processes are also
          # writing to, so let's not try to create any files there because we
          # cannot generate safe temp file names. Instead just check that the
          # directory exists. It'd be nice to check that the directory is
          # writable, but I (@laszlocsomor) don't know how to do that without
          # actually attempting to write to the directory.
          'type "%input_file:/=\\%" > "%1"',
          'if exist "%TMP%" (echo TMP:y >> "%1") else (echo TMP:n >> "%1")',
          'if exist "%TEMP%" (echo TEMP:y >> "%1") else (echo TEMP:n >> "%1")',
          'set TMP >> "%1"',
          'set TEMP >> "%1"',
          'exit /B 0',
      ]
    else:
      toolname = 'foo.sh'
      toolsrc = [
          '#!/bin/bash',
          'set -eu',
          'if [ -n "${TMPDIR:-}" ]; then',
          '  sleep 1',
          '  cat "$2" > "$1"',
          # TMPDIR might be "/tmp" or other shared directory, so we need a
          # unique name for the temp file we want to create there.
          '  tmpfile="$(mktemp "$TMPDIR/tmp.XXXXXXXX")"',
          '  echo foo > "$tmpfile"',
          '  cat "$tmpfile" >> "$1"',
          '  rm "$tmpfile"',
          '  echo "TMPDIR=${TMPDIR}" >> "$1"',
          'else',
          '  exit 1',
          'fi',
      ]

    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/' + toolname, toolsrc, executable=True)
    self.ScratchFile(
        'foo/foo.bzl',
        [
            'def _impl(ctx):',
            '  ctx.actions.run(',
            '      executable=ctx.executable.tool,',
            '      arguments=[ctx.outputs.out.path, ctx.file.src.path],',
            '      inputs=[ctx.file.src],',
            '      outputs=[ctx.outputs.out])',
            '  return [DefaultInfo(files=depset([ctx.outputs.out]))]',
            '',
            'foorule = rule(',
            '    implementation=_impl,',
            '    attrs={"tool": attr.label(executable=True, cfg="exec",',
            '                              allow_single_file=True),',
            '           "src": attr.label(allow_single_file=True)},',
            '    outputs={"out": "%{name}.txt"},',
            ')',
        ],
    )

    self.ScratchFile('foo/BUILD', [
        'load("//foo:foo.bzl", "foorule")',
        '',
        'genrule(',
        '    name = "genrule",',
        '    tools = ["%s"],' % toolname,
        '    srcs = ["input.txt"],',
        '    outs = ["genrule.txt"],',
        '    cmd = "$(location %s) $@ $(location input.txt)",' % toolname,
        ')',
        '',
        'foorule(',
        '    name = "starlark",',
        '    src = "input.txt",',
        '    tool = "%s",' % toolname,
        ')',
    ])

  def _SpawnStrategies(self):
    """Returns the list of supported --spawn_strategy values."""
    exit_code, _, stderr = self.RunBazel(
        ['build', '--color=no', '--curses=no', '--spawn_strategy=foo'],
        allow_failure=True,
    )
    self.AssertExitCode(exit_code, 2, stderr)
    pattern = re.compile(r'^ERROR:.*no strategy.*Valid values are: \[(.*)\]$')
    for line in stderr:
      m = pattern.match(line)
      if m:
        return set(e.strip() for e in m.groups()[0].split(','))
    return []

  def _BuildRules(self,
                  strategy,
                  bazel_bin,
                  bazel_genfiles,
                  env_remove=None,
                  env_add=None):

    def _ReadFile(path):
      with open(path, 'rt') as f:
        return [l.strip() for l in f]

    self.RunBazel(
        [
            'build',
            '--verbose_failures',
            '--spawn_strategy=%s' % strategy,
            '//foo:genrule',
            '//foo:starlark',
        ],
        env_remove,
        env_add,
    )
    self.assertTrue(
        os.path.exists(os.path.join(bazel_genfiles, 'foo/genrule.txt')))
    self.assertTrue(os.path.exists(os.path.join(bazel_bin, 'foo/starlark.txt')))

    return {
        'genrule': _ReadFile(os.path.join(bazel_genfiles, 'foo/genrule.txt')),
        'starlark': _ReadFile(os.path.join(bazel_bin, 'foo/starlark.txt'))
    }

  def _AssertOutputFileContents(self, lines, input_file_line,
                                expected_tmpdir_regex):
    if test_base.TestBase.IsWindows():
      # 5 lines = input_file_line, TMP:y, TEMP:y, TMP=<path>, TEMP=<path>
      if len(lines) != 5:
        self.fail('lines=%s' % lines)
      self.assertEqual(lines[0:3], [input_file_line, 'TMP:y', 'TEMP:y'])
      tmp = lines[3].split('=', 1)[1]
      temp = lines[4].split('=', 1)[1]
      self.assertRegexpMatches(tmp, expected_tmpdir_regex)
      self.assertEqual(tmp, temp)
    else:
      # 3 lines = input_file_line, foo, TMPDIR
      if len(lines) != 3:
        self.fail('lines=%s' % lines)
      self.assertEqual(lines[0:2], [input_file_line, 'foo'])
      tmpdir = lines[2].split('=', 1)[1]
      self.assertRegexpMatches(tmpdir, expected_tmpdir_regex)


if __name__ == '__main__':
  unittest.main()
