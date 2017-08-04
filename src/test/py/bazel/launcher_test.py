# pylint: disable=g-bad-file-header
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
import stat
import unittest
from src.test.py.bazel import test_base


class LauncherTest(test_base.TestBase):

  def testJavaBinaryLauncher(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'java_binary(',
        '  name = "foo",',
        '  srcs = ["Main.java"],',
        '  main_class = "Main",',
        '  data = ["//bar:bar.txt"],',
        ')',
    ])
    self.ScratchFile('foo/Main.java', [
        'public class Main {',
        '  public static void main(String[] args) {'
        '    System.out.println("hello java");',
        '    System.out.println("java_runfiles=" + ',
        '        System.getenv("JAVA_RUNFILES"));',
        '    System.out.println("runfiles_manifest_only=" + ',
        '        System.getenv("RUNFILES_MANIFEST_ONLY"));',
        '    System.out.println("runfiles_manifest_file=" + ',
        '        System.getenv("RUNFILES_MANIFEST_FILE"));',
        '  }',
        '}',
    ])
    self.ScratchFile('bar/BUILD', ['exports_files(["bar.txt"])'])
    self.ScratchFile('bar/bar.txt', ['hello'])

    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-bin'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    exit_code, _, stderr = self.RunBazel(['build', '//foo'])
    self.AssertExitCode(exit_code, 0, stderr)
    main_binary = os.path.join(bazel_bin,
                               'foo/foo%s' % ('.cmd'
                                              if self.IsWindows() else ''))
    self.assertTrue(os.path.isfile(main_binary))
    self.assertTrue(os.path.isdir(os.path.join(bazel_bin, 'foo/foo.runfiles')))

    if self.IsWindows():
      self.assertTrue(os.path.isfile(main_binary))
      self.AssertRunfilesManifestContains(
          os.path.join(bazel_bin, 'foo/foo.runfiles/MANIFEST'),
          '__main__/bar/bar.txt')
    else:
      self.assertTrue(
          os.path.islink(
              os.path.join(bazel_bin, 'foo/foo.runfiles/__main__/bar/bar.txt')))

    exit_code, stdout, stderr = self.RunProgram([main_binary])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertEqual(len(stdout), 4)
    self.assertEqual(stdout[0], 'hello java')
    if self.IsWindows():
      self.assertRegexpMatches(stdout[1], r'java_runfiles=.*foo\\foo.runfiles')
      self.assertEqual(stdout[2], 'runfiles_manifest_only=1')
      self.assertRegexpMatches(
          stdout[3], r'^runfiles_manifest_file=[a-zA-Z]:[/\\].*MANIFEST$')
    else:
      self.assertRegexpMatches(stdout[1], r'java_runfiles=.*/foo/foo.runfiles')
      self.assertEqual(stdout[2], 'runfiles_manifest_only=')
      self.assertRegexpMatches(stdout[3], r'^runfiles_manifest_file.*MANIFEST$')

  def _buildShBinaryTargets(self, bazel_bin, launcher_flag, bin1_suffix):
    exit_code, _, stderr = self.RunBazel(['build', '//foo:bin1.sh'] +
                                         launcher_flag)
    self.AssertExitCode(exit_code, 0, stderr)

    bin1 = os.path.join(bazel_bin, 'foo', 'bin1.sh.%s' % bin1_suffix
                        if self.IsWindows() else 'bin1.sh')

    self.assertTrue(os.path.exists(bin1))
    self.assertTrue(
        os.path.isdir(os.path.join(bazel_bin, 'foo/bin1.sh.runfiles')))

    exit_code, _, stderr = self.RunBazel(['build', '//foo:bin2.cmd'] +
                                         launcher_flag)
    self.AssertExitCode(exit_code, 0, stderr)

    bin2 = os.path.join(bazel_bin, 'foo/bin2.cmd')
    self.assertTrue(os.path.exists(bin2))
    self.assertTrue(
        os.path.isdir(os.path.join(bazel_bin, 'foo/bin2.cmd.runfiles')))

    exit_code, _, stderr = self.RunBazel(['build', '//foo:bin3.bat'] +
                                         launcher_flag)
    if self.IsWindows():
      self.AssertExitCode(exit_code, 1, stderr)
      self.assertIn('target name extension should match source file extension.',
                    os.linesep.join(stderr))
    else:
      bin3 = os.path.join(bazel_bin, 'foo', 'bin3.bat')
      self.assertTrue(os.path.exists(bin3))
      self.assertTrue(
          os.path.isdir(os.path.join(bazel_bin, 'foo/bin3.bat.runfiles')))

    if self.IsWindows():
      self.assertTrue(os.path.isfile(bin1))
      self.assertTrue(os.path.isfile(bin2))
    else:
      self.assertTrue(os.path.islink(bin1))
      self.assertTrue(os.path.islink(bin2))
      self.assertTrue(os.path.islink(bin3))

    if self.IsWindows():
      self.AssertRunfilesManifestContains(
          os.path.join(bazel_bin, 'foo/bin1.sh.runfiles/MANIFEST'),
          '__main__/bar/bar.txt')
      self.AssertRunfilesManifestContains(
          os.path.join(bazel_bin, 'foo/bin2.cmd.runfiles/MANIFEST'),
          '__main__/bar/bar.txt')
    else:
      self.assertTrue(
          os.path.islink(
              os.path.join(bazel_bin,
                           'foo/bin1.sh.runfiles/__main__/bar/bar.txt')))
      self.assertTrue(
          os.path.islink(
              os.path.join(bazel_bin,
                           'foo/bin2.cmd.runfiles/__main__/bar/bar.txt')))
      self.assertTrue(
          os.path.islink(
              os.path.join(bazel_bin,
                           'foo/bin3.bat.runfiles/__main__/bar/bar.txt')))

    exit_code, stdout, stderr = self.RunProgram([bin1])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertEqual(len(stdout), 3)
    self.assertEqual(stdout[0], 'hello shell')
    if self.IsWindows():
      self.assertEqual(stdout[1], 'runfiles_manifest_only=1')
      self.assertRegexpMatches(stdout[2],
                               (r'^runfiles_manifest_file='
                                r'[a-zA-Z]:/.*/foo/bin1.sh.runfiles/MANIFEST$'))
    else:
      # TODO(laszlocsomor): Find out whether the runfiles-related envvars should
      # be set on Linux (e.g. $RUNFILES, $RUNFILES_MANIFEST_FILE). Currently
      # they aren't, and that may be a bug. If it's indeed a bug, fix that bug
      # and update this test.
      self.assertEqual(stdout[1], 'runfiles_manifest_only=')
      self.assertEqual(stdout[2], 'runfiles_manifest_file=')

    if self.IsWindows():
      exit_code, stdout, stderr = self.RunProgram([bin2])
      self.AssertExitCode(exit_code, 0, stderr)
      self.assertEqual(stdout[0], 'hello batch')

  def testShBinaryLauncher(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile(
        'foo/BUILD',
        [
            # On Linux/MacOS, all sh_binary rules generate an output file with
            # the same name as the rule, and this is a symlink to the file in
            # `srcs`. (Bazel allows only one file in `sh_binary.srcs`.)
            # On Windows, if the srcs's extension is one of ".exe", ".cmd", or
            # ".bat", then Bazel requires the rule's name has the same
            # extension, and the output file will be a copy of the source file.
            'sh_binary(',
            '  name = "bin1.sh",',
            '  srcs = ["foo.sh"],',
            '  data = ["//bar:bar.txt"],',
            ')',
            'sh_binary(',
            '  name = "bin2.cmd",',  # name's extension matches that of srcs[0]
            '  srcs = ["foo.cmd"],',
            '  data = ["//bar:bar.txt"],',
            ')',
            'sh_binary(',
            '  name = "bin3.bat",',  # name's extension doesn't match srcs[0]'s
            '  srcs = ["foo.cmd"],',
            '  data = ["//bar:bar.txt"],',
            ')',
        ])
    foo_sh = self.ScratchFile('foo/foo.sh', [
        '#!/bin/bash',
        'echo hello shell',
        'echo runfiles_manifest_only=${RUNFILES_MANIFEST_ONLY:-}',
        'echo runfiles_manifest_file=${RUNFILES_MANIFEST_FILE:-}',
    ])
    foo_cmd = self.ScratchFile('foo/foo.cmd', ['@echo hello batch'])
    self.ScratchFile('bar/BUILD', ['exports_files(["bar.txt"])'])
    self.ScratchFile('bar/bar.txt', ['hello'])
    os.chmod(foo_sh, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    os.chmod(foo_cmd, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-bin'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    self._buildShBinaryTargets(bazel_bin, ['--windows_exe_launcher=0'], 'cmd')
    self._buildShBinaryTargets(bazel_bin, [], 'exe')

  def testShBinaryArgumentPassing(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'sh_binary(',
        '  name = "bin",',
        '  srcs = ["bin.sh"],',
        ')',
    ])
    foo_sh = self.ScratchFile('foo/bin.sh', [
        '#!/bin/bash',
        '# Store arguments in a array',
        'args=("$@")',
        '# Get the number of arguments',
        'N=${#args[@]}',
        '# Echo each argument',
        'for (( i=0;i<$N;i++)); do',
        ' echo ${args[${i}]}',
        'done',
    ])
    os.chmod(foo_sh, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-bin'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    exit_code, _, stderr = self.RunBazel(
        ['build', '--windows_exe_launcher', '//foo:bin'])
    self.AssertExitCode(exit_code, 0, stderr)

    bin1 = os.path.join(bazel_bin, 'foo', 'bin.exe'
                        if self.IsWindows() else 'bin')
    self.assertTrue(os.path.exists(bin1))
    self.assertTrue(os.path.isdir(os.path.join(bazel_bin, 'foo/bin.runfiles')))

    arguments = ['a', 'a b', '"b"', 'C:\\a\\b\\', '"C:\\a b\\c\\"']
    exit_code, stdout, stderr = self.RunProgram([bin1] + arguments)
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertEqual(stdout, arguments)

  def testPyBinaryLauncher(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/foo.bzl', [
        'def _impl(ctx):',
        '  ctx.actions.run(',
        '      arguments=[ctx.outputs.out.path],',
        '      outputs=[ctx.outputs.out],',
        '      executable=ctx.executable._hello_world,',
        '      use_default_shell_env=True)',
        '',
        'helloworld = rule(',
        '  implementation=_impl,',
        '  attrs={',
        '      "srcs": attr.label_list(allow_files=True),',
        '      "out": attr.output(mandatory=True),',
        '      "_hello_world": attr.label(executable=True, cfg="host",',
        '                                 allow_files=True,',
        '                                 default=Label("//foo:foo"))',
        '  }',
        ')',
    ])
    self.ScratchFile('foo/BUILD', [
        'load(":foo.bzl", "helloworld")',
        '',
        'py_binary(',
        '  name = "foo",',
        '  srcs = ["foo.py"],',
        '  data = ["//bar:bar.txt"],',
        ')',
        '',
        'helloworld(',
        '  name = "hello",',
        '  out = "hello.txt",',
        ')'
    ])
    foo_py = self.ScratchFile('foo/foo.py', [
        '#!/usr/bin/env python',
        'import sys',
        'if len(sys.argv) == 2:',
        '  with open(sys.argv[1], "w") as f:',
        '    f.write("Hello World!")',
        'else:',
        '  print("Hello World!")',
    ])
    self.ScratchFile('bar/BUILD', ['exports_files(["bar.txt"])'])
    self.ScratchFile('bar/bar.txt', ['hello'])
    os.chmod(foo_py, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-bin'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    # Verify that the build of our py_binary succeeds.
    exit_code, _, stderr = self.RunBazel(['build', '//foo:foo'])
    self.AssertExitCode(exit_code, 0, stderr)

    # Verify that generated files exist.
    foo_bin = os.path.join(bazel_bin, 'foo', 'foo.cmd'
                           if self.IsWindows() else 'foo')
    self.assertTrue(os.path.isfile(foo_bin))
    self.assertTrue(os.path.isdir(os.path.join(bazel_bin, 'foo/foo.runfiles')))

    # Verify contents of runfiles (manifest).
    if self.IsWindows():
      self.AssertRunfilesManifestContains(
          os.path.join(bazel_bin, 'foo/foo.runfiles/MANIFEST'),
          '__main__/bar/bar.txt')
    else:
      self.assertTrue(
          os.path.islink(
              os.path.join(bazel_bin, 'foo/foo.runfiles/__main__/bar/bar.txt')))

    # Try to run the built py_binary.
    exit_code, stdout, stderr = self.RunProgram([foo_bin])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertEqual(stdout[0], 'Hello World!')

    # Try to use the py_binary as an executable in a Skylark rule.
    exit_code, stdout, stderr = self.RunBazel(['build', '//foo:hello'])
    self.AssertExitCode(exit_code, 0, stderr)

    # Verify that the Skylark action generated the right output.
    hello_path = os.path.join(bazel_bin, 'foo', 'hello.txt')
    self.assertTrue(os.path.isfile(hello_path))
    with open(hello_path, 'r') as f:
      self.assertEqual(f.read(), 'Hello World!')

  def AssertRunfilesManifestContains(self, manifest, entry):
    with open(manifest, 'r') as f:
      for l in f:
        tokens = l.strip().split(' ', 1)
        if len(tokens) == 2 and tokens[0] == entry:
          return
    self.fail('Runfiles manifest "%s" did not contain "%s"' % (manifest, entry))


if __name__ == '__main__':
  unittest.main()
