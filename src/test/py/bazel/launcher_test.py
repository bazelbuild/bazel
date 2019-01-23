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

  def _buildJavaTargets(self, bazel_bin, binary_suffix):
    exit_code, _, stderr = self.RunBazel(['build', '//foo'])
    self.AssertExitCode(exit_code, 0, stderr)
    main_binary = os.path.join(bazel_bin, 'foo/foo%s' % binary_suffix)
    self.assertTrue(os.path.isfile(main_binary))
    self.assertTrue(
        os.path.isdir(
            os.path.join(bazel_bin, 'foo/foo%s.runfiles' % binary_suffix)))

    if self.IsWindows():
      self.assertTrue(os.path.isfile(main_binary))
      self.AssertRunfilesManifestContains(
          os.path.join(bazel_bin,
                       'foo/foo%s.runfiles/MANIFEST' % binary_suffix),
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
      self.assertRegexpMatches(
          stdout[1], r'java_runfiles=.*foo\\foo%s.runfiles' % binary_suffix)
      self.assertEqual(stdout[2], 'runfiles_manifest_only=1')
      self.assertRegexpMatches(
          stdout[3], r'^runfiles_manifest_file=[a-zA-Z]:[/\\].*MANIFEST$')
    else:
      self.assertRegexpMatches(stdout[1], r'java_runfiles=.*/foo/foo.runfiles')
      self.assertEqual(stdout[2], 'runfiles_manifest_only=')
      self.assertRegexpMatches(stdout[3], r'^runfiles_manifest_file.*MANIFEST$')

  def _buildShBinaryTargets(self, bazel_bin, bin1_suffix):
    exit_code, _, stderr = self.RunBazel(['build', '//foo:bin1.sh'])
    self.AssertExitCode(exit_code, 0, stderr)

    bin1 = os.path.join(bazel_bin, 'foo', 'bin1.sh%s' % bin1_suffix)

    self.assertTrue(os.path.exists(bin1))
    self.assertTrue(
        os.path.isdir(
            os.path.join(bazel_bin, 'foo/bin1.sh%s.runfiles' % bin1_suffix)))

    exit_code, _, stderr = self.RunBazel(['build', '//foo:bin2.cmd'])
    self.AssertExitCode(exit_code, 0, stderr)

    bin2 = os.path.join(bazel_bin, 'foo/bin2.cmd')
    self.assertTrue(os.path.exists(bin2))
    self.assertTrue(
        os.path.isdir(os.path.join(bazel_bin, 'foo/bin2.cmd.runfiles')))

    exit_code, _, stderr = self.RunBazel(['build', '//foo:bin3.bat'])
    if self.IsWindows():
      self.AssertExitCode(exit_code, 1, stderr)
      self.assertIn('target name extension should match source file extension',
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
          os.path.join(bazel_bin,
                       'foo/bin1.sh%s.runfiles/MANIFEST' % bin1_suffix),
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
      self.assertRegexpMatches(
          stdout[2],
          (r'^runfiles_manifest_file='
           r'[a-zA-Z]:/.*/foo/bin1.sh%s.runfiles/MANIFEST$' % bin1_suffix))
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

  def _buildPyTargets(self, bazel_bin, binary_suffix):
    # Verify that the build of our py_binary succeeds.
    exit_code, _, stderr = self.RunBazel(['build', '//foo:foo'])
    self.AssertExitCode(exit_code, 0, stderr)

    # Verify that generated files exist.
    foo_bin = os.path.join(bazel_bin, 'foo', 'foo%s' % binary_suffix)
    self.assertTrue(os.path.isfile(foo_bin))
    self.assertTrue(
        os.path.isdir(
            os.path.join(bazel_bin, 'foo/foo%s.runfiles' % binary_suffix)))

    # Verify contents of runfiles (manifest).
    if self.IsWindows():
      self.AssertRunfilesManifestContains(
          os.path.join(bazel_bin,
                       'foo/foo%s.runfiles/MANIFEST' % binary_suffix),
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

    # Verify that running py_test succeeds.
    exit_code, _, stderr = self.RunBazel(['test', '//foo:test'])
    self.AssertExitCode(exit_code, 0, stderr)

  def _buildAndCheckArgumentPassing(self, package, target_name):
    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-bin'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    exit_code, _, stderr = self.RunBazel(
        ['build', '//%s:%s' % (package, target_name)])
    self.AssertExitCode(exit_code, 0, stderr)

    bin_suffix = '.exe' if self.IsWindows() else ''
    bin1 = os.path.join(bazel_bin, package, '%s%s' % (target_name, bin_suffix))
    self.assertTrue(os.path.exists(bin1))
    self.assertTrue(
        os.path.isdir(
            os.path.join(bazel_bin, '%s/%s%s.runfiles' % (package, target_name,
                                                          bin_suffix))))

    arguments = ['a', 'a b', '"b"', 'C:\\a\\b\\', '"C:\\a b\\c\\"']
    exit_code, stdout, stderr = self.RunProgram([bin1] + arguments)
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertEqual(stdout, arguments)

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
    self._buildJavaTargets(bazel_bin, '.exe' if self.IsWindows() else '')

  def testJavaBinaryArgumentPassing(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'java_binary(',
        '  name = "bin",',
        '  srcs = ["Main.java"],',
        '  main_class = "Main",',
        ')',
    ])
    self.ScratchFile('foo/Main.java', [
        'public class Main {',
        '  public static void main(String[] args) {'
        '    for (String arg : args) {',
        '      System.out.println(arg);',
        '    }'
        '  }',
        '}',
    ])

    self._buildAndCheckArgumentPassing('foo', 'bin')

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
    self._buildShBinaryTargets(bazel_bin, '.exe' if self.IsWindows() else '')

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

    self._buildAndCheckArgumentPassing('foo', 'bin')

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
        'load(":foo.bzl", "helloworld")', '', 'py_binary(', '  name = "foo",',
        '  srcs = ["foo.py"],', '  data = ["//bar:bar.txt"],', ')', '',
        'py_test(', '  name = "test",', '  srcs = ["test.py"],', ')', '',
        'helloworld(', '  name = "hello",', '  out = "hello.txt",', ')'
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
    test_py = self.ScratchFile('foo/test.py', [
        '#!/usr/bin/env python',
        'import unittest',
        'class MyTest(unittest.TestCase):',
        '  def test_dummy(self):',
        '      pass',
        'if __name__ == \'__main__\':',
        '  unittest.main()',
    ])
    self.ScratchFile('bar/BUILD', ['exports_files(["bar.txt"])'])
    self.ScratchFile('bar/bar.txt', ['hello'])
    os.chmod(foo_py, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    os.chmod(test_py, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-bin'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]
    self._buildPyTargets(bazel_bin, '.exe' if self.IsWindows() else '')

  def testPyBinaryArgumentPassing(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'py_binary(',
        '  name = "bin",',
        '  srcs = ["bin.py"],',
        ')',
    ])
    self.ScratchFile('foo/bin.py', [
        'import sys',
        'for arg in sys.argv[1:]:',
        '  print(arg)',
    ])

    self._buildAndCheckArgumentPassing('foo', 'bin')

  def testWindowsJavaExeLauncher(self):
    # Skip this test on non-Windows platforms
    if not self.IsWindows():
      return
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'java_binary(',
        '  name = "foo",',
        '  srcs = ["Main.java"],',
        '  main_class = "Main",',
        '  jvm_flags = ["--flag1", "--flag2"],',
        ')',
    ])
    self.ScratchFile('foo/Main.java', [
        'public class Main {',
        '  public static void main(String[] args) {'
        '    System.out.println("helloworld");',
        '  }',
        '}',
    ])

    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-bin'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    exit_code, _, stderr = self.RunBazel(['build', '//foo:foo'])
    self.AssertExitCode(exit_code, 0, stderr)

    binary = os.path.join(bazel_bin, 'foo', 'foo.exe')
    self.assertTrue(os.path.exists(binary))

    # Add this flag to make launcher print the command it generated instead of
    # launching the real program.
    print_cmd = '--print_launcher_command'

    exit_code, stdout, stderr = self.RunProgram([binary, '--debug', print_cmd])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn(
        '-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005',
        stdout)

    exit_code, stdout, stderr = self.RunProgram(
        [binary, '--debug', print_cmd],
        env_add={'DEFAULT_JVM_DEBUG_PORT': '12345'})
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn(
        '-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=12345',
        stdout)

    exit_code, stdout, stderr = self.RunProgram(
        [binary, '--debug=12345', print_cmd],
        env_add={
            'DEFAULT_JVM_DEBUG_SUSPEND': 'n',
            'PERSISTENT_TEST_RUNNER': 'true'
        })
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn(
        '-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=12345'
        ',quiet=y', stdout)

    exit_code, stdout, stderr = self.RunProgram(
        [binary, '--main_advice=MyMain', print_cmd])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn('MyMain', stdout)

    exit_code, stdout, stderr = self.RunProgram(
        [binary, '--main_advice_classpath=foo/bar', print_cmd])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn('-classpath', stdout)
    classpath = stdout[stdout.index('-classpath') + 1]
    self.assertIn('foo/bar', classpath)

    exit_code, stdout, stderr = self.RunProgram(
        [binary, '--jvm_flag="--some_path="./a b/c""', print_cmd])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn('"--some_path=\\"./a b/c\\""', stdout)

    exit_code, stdout, stderr = self.RunProgram(
        [binary, '--jvm_flags="--path1=a --path2=b"', print_cmd])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn('--path1=a', stdout)
    self.assertIn('--path2=b', stdout)

    exit_code, stdout, stderr = self.RunProgram(
        [binary, print_cmd], env_add={'JVM_FLAGS': '--foo --bar'})
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn('--flag1', stdout)
    self.assertIn('--flag2', stdout)
    self.assertIn('--foo', stdout)
    self.assertIn('--bar', stdout)

    exit_code, stdout, stderr = self.RunProgram(
        [binary, '--singlejar', print_cmd])
    self.AssertExitCode(exit_code, 1, stderr)
    self.assertIn('foo_deploy.jar does not exist', ''.join(stderr))
    exit_code, _, stderr = self.RunBazel(['build', '//foo:foo_deploy.jar'])
    self.AssertExitCode(exit_code, 0, stderr)
    exit_code, stdout, stderr = self.RunProgram(
        [binary, '--singlejar', print_cmd])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn('-classpath', stdout)
    classpath = stdout[stdout.index('-classpath') + 1]
    self.assertIn('foo_deploy.jar', classpath)

    exit_code, stdout, stderr = self.RunProgram([binary, '--print_javabin'])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn('local_jdk/bin/java.exe', ''.join(stdout))

    my_tmp_dir = self.ScratchDir('my/temp/dir')
    exit_code, stdout, stderr = self.RunProgram(
        [binary, print_cmd], env_add={'TEST_TMPDIR': my_tmp_dir})
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn('-Djava.io.tmpdir=%s' % my_tmp_dir, stdout)

    exit_code, stdout, stderr = self.RunProgram(
        [binary, '--classpath_limit=0', print_cmd])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn('-classpath', stdout)
    classpath = stdout[stdout.index('-classpath') + 1]
    self.assertRegexpMatches(classpath, r'foo-[A-Za-z0-9]+-classpath.jar$')

  def testWindowsNativeLauncherInNonEnglishPath(self):
    if not self.IsWindows():
      return
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('bin/BUILD', [
        'java_binary(',
        '  name = "bin_java",',
        '  srcs = ["Main.java"],',
        '  main_class = "Main",',
        ')',
        'sh_binary(',
        '  name = "bin_sh",',
        '  srcs = ["main.sh"],',
        ')',
    ])
    self.ScratchFile('bin/Main.java', [
        'public class Main {',
        '  public static void main(String[] args) {'
        '    System.out.println("helloworld");',
        '  }',
        '}',
    ])
    self.ScratchFile('bin/main.sh', [
        'echo "helloworld"',
    ])

    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-bin'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    exit_code, _, stderr = self.RunBazel(['build', '//bin/...'])
    self.AssertExitCode(exit_code, 0, stderr)

    for f in [
        'bin_java.exe', 'bin_java.exe.runfiles_manifest',
        'bin_sh.exe', 'bin_sh', 'bin_sh.exe.runfiles_manifest',
    ]:
      self.CopyFile(os.path.join(bazel_bin, 'bin', f),
                    os.path.join(u'./\u6d4b\u8bd5', f))

    unicode_binary_path = u'./\u6d4b\u8bd5/bin_java.exe'
    exit_code, stdout, stderr = self.RunProgram([unicode_binary_path])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertEqual('helloworld', ''.join(stdout))

    unicode_binary_path = u'./\u6d4b\u8bd5/bin_sh.exe'
    exit_code, stdout, stderr = self.RunProgram([unicode_binary_path])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertEqual('helloworld', ''.join(stdout))

  def testWindowsNativeLauncherInLongPath(self):
    if not self.IsWindows():
      return
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('bin/BUILD', [
        'java_binary(',
        '  name = "bin_java",',
        '  srcs = ["Main.java"],',
        '  main_class = "Main",',
        ')',
        'sh_binary(',
        '  name = "bin_sh",',
        '  srcs = ["main.sh"],',
        ')',
        'py_binary(',
        '  name = "bin_py",',
        '  srcs = ["bin_py.py"],',
        ')',
    ])
    self.ScratchFile('bin/Main.java', [
        'public class Main {',
        '  public static void main(String[] args) {'
        '    System.out.println("helloworld");',
        '  }',
        '}',
    ])
    self.ScratchFile('bin/main.sh', [
        'echo "helloworld"',
    ])
    self.ScratchFile('bin/bin_py.py', [
        'print("helloworld")',
    ])

    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-bin'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    exit_code, _, stderr = self.RunBazel(['build', '//bin/...'])
    self.AssertExitCode(exit_code, 0, stderr)

    # Create a directory with a path longer than 260
    long_dir_path = "./" + "/".join(["a" * 100, "b" * 100, "c" * 100])

    for f in [
        'bin_java.exe', 'bin_java.exe.runfiles_manifest',
        'bin_sh.exe', 'bin_sh', 'bin_sh.exe.runfiles_manifest',
        'bin_py.exe', 'bin_py.zip', 'bin_py.exe.runfiles_manifest',
    ]:
      self.CopyFile(os.path.join(bazel_bin, 'bin', f),
                    os.path.join(long_dir_path, f))

    long_binary_path = os.path.abspath(long_dir_path + '/bin_java.exe')
    # To run a binary at a long path, we need to set shell=True
    exit_code, stdout, stderr = self.RunProgram([long_binary_path], shell=True)
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertEqual('helloworld', ''.join(stdout))

    long_binary_path = os.path.abspath(long_dir_path + '/bin_sh.exe')
    # To run a binary at a long path, we need to set shell=True
    exit_code, stdout, stderr = self.RunProgram([long_binary_path], shell=True)
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertEqual('helloworld', ''.join(stdout))

    long_binary_path = os.path.abspath(long_dir_path + '/bin_py.exe')
    # To run a binary at a long path, we need to set shell=True
    exit_code, stdout, stderr = self.RunProgram([long_binary_path], shell=True)
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertEqual('helloworld', ''.join(stdout))

  def AssertRunfilesManifestContains(self, manifest, entry):
    with open(manifest, 'r') as f:
      for l in f:
        tokens = l.strip().split(' ', 1)
        if len(tokens) == 2 and tokens[0] == entry:
          return
    self.fail('Runfiles manifest "%s" did not contain "%s"' % (manifest, entry))


if __name__ == '__main__':
  unittest.main()
