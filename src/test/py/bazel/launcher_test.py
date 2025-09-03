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
import string
from absl.testing import absltest
from src.test.py.bazel import test_base

# pylint: disable=g-import-not-at-top
if os.name == 'nt':
  import ctypes
  from ctypes import wintypes

  kernel32 = ctypes.WinDLL('kernel32')
  _GetShortPathNameW = kernel32.GetShortPathNameW
  _GetShortPathNameW.argtypes = [
      wintypes.LPCWSTR,
      wintypes.LPWSTR,
      wintypes.DWORD,
  ]
  _GetShortPathNameW.restype = wintypes.DWORD

  def _get_short_path_name(long_name):
    # Gets the short path name of a given long path.
    # http://stackoverflow.com/a/23598461/200291

    output_buf_size = len(long_name)
    while True:
      output_buf = ctypes.create_unicode_buffer(output_buf_size)
      needed = _GetShortPathNameW(long_name, output_buf, output_buf_size)
      if needed == 0:
        raise ctypes.WinError()
      elif output_buf_size >= needed:
        return output_buf.value
      else:
        output_buf_size = needed


class LauncherTest(test_base.TestBase):

  def _buildJavaTargets(self, bazel_bin, binary_suffix):
    self.RunBazel(['build', '//foo'])
    main_binary = os.path.join(bazel_bin, 'foo/foo%s' % binary_suffix)
    self.assertTrue(os.path.isfile(main_binary))
    self.assertTrue(
        os.path.isdir(
            os.path.join(bazel_bin, 'foo/foo%s.runfiles' % binary_suffix)))

    if self.IsWindows():
      self.assertTrue(os.path.isfile(main_binary))
      self.AssertRunfilesManifestContains(
          os.path.join(
              bazel_bin, 'foo/foo%s.runfiles/MANIFEST' % binary_suffix
          ),
          '_main/bar/bar.txt',
      )
    else:
      self.assertTrue(
          os.path.islink(
              os.path.join(bazel_bin, 'foo/foo.runfiles/_main/bar/bar.txt')
          )
      )

    _, stdout, _ = self.RunProgram([main_binary])
    self.assertEqual(len(stdout), 4)
    self.assertEqual(stdout[0], 'hello java')
    if self.IsWindows():
      self.assertRegex(
          stdout[1], r'java_runfiles=.*foo\\foo%s.runfiles' % binary_suffix)
      self.assertEqual(stdout[2], 'runfiles_manifest_only=1')
      self.assertRegex(
          stdout[3], r'^runfiles_manifest_file=[a-zA-Z]:[/\\].*MANIFEST$')
    else:
      self.assertRegex(stdout[1], r'java_runfiles=.*/foo/foo.runfiles')
      self.assertEqual(stdout[2], 'runfiles_manifest_only=')
      self.assertRegex(stdout[3], r'^runfiles_manifest_file.*MANIFEST$')

  def _buildShBinaryTargets(self, bazel_bin, bin1_suffix):
    self.RunBazel(['build', '//foo:bin1.sh'])

    bin1 = os.path.join(bazel_bin, 'foo', 'bin1.sh%s' % bin1_suffix)

    self.assertTrue(os.path.exists(bin1))
    self.assertTrue(
        os.path.isdir(
            os.path.join(bazel_bin, 'foo/bin1.sh%s.runfiles' % bin1_suffix)))

    self.RunBazel(['build', '//foo:bin2.cmd'])

    bin2 = os.path.join(bazel_bin, 'foo/bin2.cmd')
    self.assertTrue(os.path.exists(bin2))
    self.assertTrue(
        os.path.isdir(os.path.join(bazel_bin, 'foo/bin2.cmd.runfiles')))

    exit_code, _, stderr = self.RunBazel(
        ['build', '//foo:bin3.bat'], allow_failure=True
    )
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
          os.path.join(
              bazel_bin, 'foo/bin1.sh%s.runfiles/MANIFEST' % bin1_suffix
          ),
          '_main/bar/bar.txt',
      )
      self.AssertRunfilesManifestContains(
          os.path.join(bazel_bin, 'foo/bin2.cmd.runfiles/MANIFEST'),
          '_main/bar/bar.txt',
      )
    else:
      self.assertTrue(
          os.path.islink(
              os.path.join(bazel_bin, 'foo/bin1.sh.runfiles/_main/bar/bar.txt')
          )
      )
      self.assertTrue(
          os.path.islink(
              os.path.join(bazel_bin, 'foo/bin2.cmd.runfiles/_main/bar/bar.txt')
          )
      )
      self.assertTrue(
          os.path.islink(
              os.path.join(bazel_bin, 'foo/bin3.bat.runfiles/_main/bar/bar.txt')
          )
      )

    _, stdout, _ = self.RunProgram([bin1])
    self.assertEqual(len(stdout), 3)
    self.assertEqual(stdout[0], 'hello shell')
    if self.IsWindows():
      self.assertEqual(stdout[1], 'runfiles_manifest_only=1')
      self.assertRegex(
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
    self.RunBazel(['build', '//foo:foo'])

    # Verify that generated files exist.
    foo_bin = os.path.join(bazel_bin, 'foo', 'foo%s' % binary_suffix)
    self.assertTrue(os.path.isfile(foo_bin))
    self.assertTrue(
        os.path.isdir(
            os.path.join(bazel_bin, 'foo/foo%s.runfiles' % binary_suffix)))

    # Verify contents of runfiles (manifest).
    if self.IsWindows():
      self.AssertRunfilesManifestContains(
          os.path.join(
              bazel_bin, 'foo/foo%s.runfiles/MANIFEST' % binary_suffix
          ),
          '_main/bar/bar.txt',
      )
    else:
      self.assertTrue(
          os.path.islink(
              os.path.join(bazel_bin, 'foo/foo.runfiles/_main/bar/bar.txt')
          )
      )

    # Try to run the built py_binary.
    _, stdout, _ = self.RunProgram([foo_bin])
    self.assertEqual(stdout[0], 'Hello World!')

    # Try to use the py_binary as an executable in a Starlark rule.
    self.RunBazel(['build', '//foo:hello'])

    # Verify that the Starlark action generated the right output.
    hello_path = os.path.join(bazel_bin, 'foo', 'hello.txt')
    self.assertTrue(os.path.isfile(hello_path))
    with open(hello_path, 'r') as f:
      self.assertEqual(f.read(), 'Hello World!')

    # Verify that running py_test succeeds.
    self.RunBazel(['test', '//foo:test'])

  def _buildAndCheckArgumentPassing(self, package, target_name):
    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]

    self.RunBazel(['build', '//%s:%s' % (package, target_name)])

    bin_suffix = '.exe' if self.IsWindows() else ''
    bin1 = os.path.join(bazel_bin, package, '%s%s' % (target_name, bin_suffix))
    self.assertTrue(os.path.exists(bin1))
    self.assertTrue(
        os.path.isdir(
            os.path.join(bazel_bin, '%s/%s%s.runfiles' % (package, target_name,
                                                          bin_suffix))))

    arguments = ['a', 'a b', '"b"', 'C:\\a\\b\\', '"C:\\a b\\c\\"']
    _, stdout, _ = self.RunProgram([bin1] + arguments)
    self.assertEqual(stdout, arguments)

  def testJavaBinaryLauncher(self):
    self.AddBazelDep('rules_java')
    self.ScratchFile(
        'foo/BUILD',
        [
            'load("@rules_java//java:java_binary.bzl", "java_binary")',
            'java_binary(',
            '  name = "foo",',
            '  srcs = ["Main.java"],',
            '  main_class = "Main",',
            '  data = ["//bar:bar.txt"],',
            ')',
        ],
    )
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

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]
    self._buildJavaTargets(bazel_bin, '.exe' if self.IsWindows() else '')

  def testJavaBinaryArgumentPassing(self):
    self.AddBazelDep('rules_java')
    self.ScratchFile(
        'foo/BUILD',
        [
            'load("@rules_java//java:java_binary.bzl", "java_binary")',
            'java_binary(',
            '  name = "bin",',
            '  srcs = ["Main.java"],',
            '  main_class = "Main",',
            ')',
        ],
    )
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
    self.AddBazelDep('rules_shell')
    self.ScratchFile(
        'foo/BUILD',
        [
            # On Linux/MacOS, all sh_binary rules generate an output file with
            # the same name as the rule, and this is a symlink to the file in
            # `srcs`. (Bazel allows only one file in `sh_binary.srcs`.)
            # On Windows, if the srcs's extension is one of ".exe", ".cmd", or
            # ".bat", then Bazel requires the rule's name has the same
            # extension, and the output file will be a copy of the source file.
            'load("@rules_shell//shell:sh_binary.bzl", "sh_binary")',
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
        ],
    )
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

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]
    self._buildShBinaryTargets(bazel_bin, '.exe' if self.IsWindows() else '')

  def testShBinaryArgumentPassing(self):
    self.AddBazelDep('rules_shell')
    self.ScratchFile(
        'foo/BUILD',
        [
            'load("@rules_shell//shell:sh_binary.bzl", "sh_binary")',
            'sh_binary(',
            '  name = "bin",',
            '  srcs = ["bin.sh"],',
            ')',
        ],
    )
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
    self.AddBazelDep('rules_python')
    self.ScratchFile(
        'foo/foo.bzl',
        [
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
            '      "_hello_world": attr.label(executable=True, cfg="exec",',
            '                                 allow_files=True,',
            '                                 default=Label("//foo:foo"))',
            '  }',
            ')',
        ],
    )
    self.ScratchFile(
        'foo/BUILD',
        [
            'load("@rules_python//python:py_binary.bzl", "py_binary")',
            'load("@rules_python//python:py_test.bzl", "py_test")',
            'load(":foo.bzl", "helloworld")',
            '',
            'py_binary(',
            '  name = "foo",',
            '  srcs = ["foo.py"],',
            '  data = ["//bar:bar.txt"],',
            ')',
            '',
            'py_test(',
            '  name = "test",',
            '  srcs = ["test.py"],',
            ')',
            '',
            'helloworld(',
            '  name = "hello",',
            '  out = "hello.txt",',
            ')',
        ],
    )
    foo_py = self.ScratchFile('foo/foo.py', [
        '#!/usr/bin/env python3',
        'import sys',
        'if len(sys.argv) == 2:',
        '  with open(sys.argv[1], "w") as f:',
        '    f.write("Hello World!")',
        'else:',
        '  print("Hello World!")',
    ])
    test_py = self.ScratchFile('foo/test.py', [
        '#!/usr/bin/env python3',
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

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]
    self._buildPyTargets(bazel_bin, '.exe' if self.IsWindows() else '')

  def testPyBinaryArgumentPassing(self):
    self.AddBazelDep('rules_python')
    self.ScratchFile(
        'foo/BUILD',
        [
            'load("@rules_python//python:py_binary.bzl", "py_binary")',
            'py_binary(',
            '  name = "bin",',
            '  srcs = ["bin.py"],',
            ')',
        ],
    )
    self.ScratchFile('foo/bin.py', [
        'import sys',
        'for arg in sys.argv[1:]:',
        '  print(arg)',
    ])

    self._buildAndCheckArgumentPassing('foo', 'bin')

  def testPyBinaryLauncherWithDifferentArgv0(self):
    """Test for https://github.com/bazelbuild/bazel/issues/14343."""
    self.AddBazelDep('rules_python')
    self.ScratchFile(
        'foo/BUILD',
        [
            'load("@rules_python//python:py_binary.bzl", "py_binary")',
            'py_binary(',
            '  name = "bin",',
            '  srcs = ["bin.py"],',
            ')',
        ],
    )
    self.ScratchFile('foo/bin.py', ['print("Hello world")'])

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]

    # Verify that the build of our py_binary succeeds.
    self.RunBazel(['build', '//foo:bin'])

    # Try to run the built py_binary.
    binary_suffix = '.exe' if self.IsWindows() else ''
    foo_bin = os.path.join(bazel_bin, 'foo', 'bin%s' % binary_suffix)
    args = [r'C:\Invalid.exe' if self.IsWindows() else '/invalid']
    _, stdout, _ = self.RunProgram(args, executable=foo_bin)
    self.assertEqual(stdout[0], 'Hello world')

  def testWindowsJavaExeLauncher(self):
    # Skip this test on non-Windows platforms
    if not self.IsWindows():
      return
    self.AddBazelDep('rules_java')
    self.ScratchFile(
        'foo/BUILD',
        [
            'load("@rules_java//java:java_binary.bzl", "java_binary")',
            'java_binary(',
            '  name = "foo",',
            '  srcs = ["Main.java"],',
            '  main_class = "Main",',
            '  jvm_flags = ["--flag1", "--flag2"],',
            '  data = ["advice-1.jar", "advice-2.jar"],',
            ')',
        ],
    )
    self.ScratchFile('foo/advice-1.jar')
    self.ScratchFile('foo/advice-2.jar')
    self.ScratchFile('foo/Main.java', [
        'public class Main {',
        '  public static void main(String[] args) {',
        '    System.out.println("helloworld");',
        '  }',
        '}',
    ])

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]

    self.RunBazel(['build', '//foo:foo'])

    binary = os.path.join(bazel_bin, 'foo', 'foo.exe')
    self.assertTrue(os.path.exists(binary))

    # Add this flag to make launcher print the command it generated instead of
    # launching the real program.
    print_cmd = '--print_launcher_command'

    _, stdout, _ = self.RunProgram([binary, '--debug', print_cmd])
    self.assertIn(
        '-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005',
        stdout)

    _, stdout, _ = self.RunProgram(
        [binary, '--debug', print_cmd],
        env_add={'DEFAULT_JVM_DEBUG_PORT': '12345'},
    )
    self.assertIn(
        '-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=12345',
        stdout)

    _, stdout, _ = self.RunProgram(
        [binary, '--debug=12345', print_cmd],
        env_add={
            'DEFAULT_JVM_DEBUG_SUSPEND': 'n',
            'PERSISTENT_TEST_RUNNER': 'true',
        },
    )
    self.assertIn(
        '-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=12345'
        ',quiet=y', stdout)

    _, stdout, _ = self.RunProgram([binary, '--main_advice=MyMain', print_cmd])
    self.assertIn('MyMain', stdout)

    _, stdout, _ = self.RunProgram([
        binary,
        '--main_advice_classpath=foo/advice-1.jar;foo/advice-2.jar',
        print_cmd,
    ])
    self.assertIn('-classpath', stdout)
    classpath = stdout[stdout.index('-classpath') + 1]
    self.assertIn('foo/advice-1.jar', classpath)
    self.assertIn('foo/advice-2.jar', classpath)

    _, stdout, _ = self.RunProgram(
        [binary, '--main_advice_classpath=C:\\foo\\bar', print_cmd]
    )
    self.assertIn('-classpath', stdout)
    classpath = stdout[stdout.index('-classpath') + 1]
    self.assertIn('C:\\foo\\bar', classpath)

    _, stdout, _ = self.RunProgram(
        [binary, '--jvm_flag="--some_path="./a b/c""', print_cmd]
    )
    self.assertIn('"--some_path=\\"./a b/c\\""', stdout)

    _, stdout, _ = self.RunProgram(
        [binary, '--jvm_flags="--path1=a --path2=b"', print_cmd]
    )
    self.assertIn('--path1=a', stdout)
    self.assertIn('--path2=b', stdout)

    _, stdout, _ = self.RunProgram(
        [binary, print_cmd], env_add={'JVM_FLAGS': '--foo --bar'}
    )
    self.assertIn('--flag1', stdout)
    self.assertIn('--flag2', stdout)
    self.assertIn('--foo', stdout)
    self.assertIn('--bar', stdout)

    exit_code, stdout, stderr = self.RunProgram(
        [binary, '--singlejar', print_cmd], allow_failure=True
    )
    self.AssertExitCode(exit_code, 1, stderr)
    self.assertIn('foo_deploy.jar does not exist', ''.join(stderr))
    self.RunBazel(['build', '//foo:foo_deploy.jar'])
    _, stdout, _ = self.RunProgram([binary, '--singlejar', print_cmd])
    self.assertIn('-classpath', stdout)
    classpath = stdout[stdout.index('-classpath') + 1]
    self.assertIn('foo_deploy.jar', classpath)

    _, stdout, _ = self.RunProgram([binary, '--print_javabin'])
    self.assertIn('local_jdk/bin/java.exe', ''.join(stdout))

    my_tmp_dir = self.ScratchDir('my/temp/dir')
    _, stdout, _ = self.RunProgram(
        [binary, print_cmd], env_add={'TEST_TMPDIR': my_tmp_dir}
    )
    self.assertIn('-Djava.io.tmpdir=%s' % my_tmp_dir, stdout)

    _, stdout, _ = self.RunProgram([binary, '--classpath_limit=0', print_cmd])
    self.assertIn('-classpath', stdout)
    classpath = stdout[stdout.index('-classpath') + 1]
    self.assertRegex(classpath, r'foo-[A-Za-z0-9]+-classpath.jar$')

  def testWindowsNativeLauncherInNonEnglishPath(self):
    if not self.IsWindows():
      return
    self.AddBazelDep('rules_java')
    self.AddBazelDep('rules_python')
    self.AddBazelDep('rules_shell')
    self.ScratchFile(
        'bin/BUILD',
        [
            'load("@rules_java//java:java_binary.bzl", "java_binary")',
            'load("@rules_shell//shell:sh_binary.bzl", "sh_binary")',
            'java_binary(',
            '  name = "bin_java",',
            '  srcs = ["Main.java"],',
            '  main_class = "Main",',
            ')',
            'sh_binary(',
            '  name = "bin_sh",',
            '  srcs = ["main.sh"],',
            ')',
        ],
    )
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

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]

    self.RunBazel(['build', '//bin/...'])

    for f in [
        'bin_java.exe',
        'bin_java.exe.runfiles_manifest',
        'bin_sh.exe',
        'bin_sh',
        'bin_sh.exe.runfiles_manifest',
    ]:
      self.CopyFile(os.path.join(bazel_bin, 'bin', f),
                    os.path.join(u'./\u6d4b\u8bd5', f))

    unicode_binary_path = u'./\u6d4b\u8bd5/bin_java.exe'
    _, stdout, _ = self.RunProgram([unicode_binary_path])
    self.assertEqual('helloworld', ''.join(stdout))

    unicode_binary_path = u'./\u6d4b\u8bd5/bin_sh.exe'
    _, stdout, _ = self.RunProgram([unicode_binary_path])
    self.assertEqual('helloworld', ''.join(stdout))

  def testWindowsNativeLauncherInLongPath(self):
    if not self.IsWindows():
      return
    self.AddBazelDep('rules_java')
    self.AddBazelDep('rules_python')
    self.AddBazelDep('rules_shell')
    self.ScratchFile(
        'bin/BUILD',
        [
            'load("@rules_java//java:java_binary.bzl", "java_binary")',
            'load("@rules_shell//shell:sh_binary.bzl", "sh_binary")',
            'load("@rules_python//python:py_binary.bzl", "py_binary")',
            'java_binary(',
            '  name = "not_short_bin_java",',
            '  srcs = ["Main.java"],',
            '  main_class = "Main",',
            ')',
            'sh_binary(',
            '  name = "not_short_bin_sh",',
            '  srcs = ["main.sh"],',
            ')',
            'py_binary(',
            '  name = "not_short_bin_py",',
            '  srcs = ["not_short_bin_py.py"],',
            ')',
        ],
    )
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
    self.ScratchFile(
        'bin/not_short_bin_py.py',
        [
            'print("helloworld")',
        ],
    )

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]

    exit_code, _, stderr = self.RunBazel(['build', '//bin/...'])
    self.AssertExitCode(exit_code, 0, stderr)

    # Create a directory with a path longer than 260
    long_dir_path = './' + '/'.join(
        [(c * 8 + '.' + c * 3) for c in string.ascii_lowercase])

    # The 'not_short_' prefix ensures that the basenames are not already 8.3
    # short paths. Due to the long directory path, the basename will thus be
    # replaced with a short path such as "not_sh~1.exe" below.
    for f in [
        'not_short_bin_java.exe',
        'not_short_bin_java.exe.runfiles_manifest',
        'not_short_bin_sh.exe',
        'not_short_bin_sh',
        'not_short_bin_sh.exe.runfiles_manifest',
        'not_short_bin_py.exe',
        'not_short_bin_py.zip',
        'not_short_bin_py.exe.runfiles_manifest',
    ]:
      self.CopyFile(
          os.path.join(bazel_bin, 'bin', f), os.path.join(long_dir_path, f))

    long_binary_path = os.path.abspath(
        long_dir_path + '/not_short_bin_java.exe'
    )
    # subprocess doesn't support long path without shell=True
    _, stdout, _ = self.RunProgram([long_binary_path], shell=True)
    self.assertEqual('helloworld', ''.join(stdout))
    # Make sure we can launch the binary with a shortened Windows 8dot3 path
    short_binary_path = _get_short_path_name(long_binary_path)
    self.assertIn('~', os.path.basename(short_binary_path))
    _, stdout, _ = self.RunProgram([short_binary_path], shell=True)
    self.assertEqual('helloworld', ''.join(stdout))

    long_binary_path = os.path.abspath(long_dir_path + '/not_short_bin_sh.exe')
    # subprocess doesn't support long path without shell=True
    _, stdout, _ = self.RunProgram([long_binary_path], shell=True)
    self.assertEqual('helloworld', ''.join(stdout))
    # Make sure we can launch the binary with a shortened Windows 8dot3 path
    short_binary_path = _get_short_path_name(long_binary_path)
    self.assertIn('~', os.path.basename(short_binary_path))
    _, stdout, _ = self.RunProgram([short_binary_path], shell=True)
    self.assertEqual('helloworld', ''.join(stdout))

    long_binary_path = os.path.abspath(long_dir_path + '/not_short_bin_py.exe')
    # subprocess doesn't support long path without shell=True
    _, stdout, _ = self.RunProgram([long_binary_path], shell=True)
    self.assertEqual('helloworld', ''.join(stdout))
    # Make sure we can launch the binary with a shortened Windows 8dot3 path
    short_binary_path = _get_short_path_name(long_binary_path)
    self.assertIn('~', os.path.basename(short_binary_path))
    _, stdout, _ = self.RunProgram([short_binary_path], shell=True)
    self.assertEqual('helloworld', ''.join(stdout))

  def testWindowsNativeLauncherInvalidArgv0(self):
    if not self.IsWindows():
      return
    self.AddBazelDep('rules_java')
    self.AddBazelDep('rules_python')
    self.AddBazelDep('rules_shell')
    self.ScratchFile(
        'bin/BUILD',
        [
            'load("@rules_java//java:java_binary.bzl", "java_binary")',
            'load("@rules_shell//shell:sh_binary.bzl", "sh_binary")',
            'load("@rules_python//python:py_binary.bzl", "py_binary")',
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
        ],
    )
    self.ScratchFile(
        'bin/Main.java',
        [
            'public class Main {',
            (
                '  public static void main(String[] args) {'
                '    System.out.println("helloworld");'
            ),
            '  }',
            '}',
        ],
    )
    self.ScratchFile(
        'bin/main.sh',
        [
            'echo "helloworld"',
        ],
    )
    self.ScratchFile(
        'bin/bin_py.py',
        [
            'print("helloworld")',
        ],
    )

    _, stdout, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = stdout[0]

    self.RunBazel(['build', '//bin/...'])

    _, stdout, _ = self.RunProgram(
        ['C:\\Invalid'],
        executable=os.path.join(bazel_bin, 'bin', 'bin_java.exe'),
    )
    self.assertEqual('helloworld', ''.join(stdout))

    _, stdout, _ = self.RunProgram(
        ['C:\\Invalid'], executable=os.path.join(bazel_bin, 'bin', 'bin_sh.exe')
    )
    self.assertEqual('helloworld', ''.join(stdout))

    _, stdout, _ = self.RunProgram(
        ['C:\\Invalid'], executable=os.path.join(bazel_bin, 'bin', 'bin_py.exe')
    )
    self.assertEqual('helloworld', ''.join(stdout))

  # Regression test for
  # https://github.com/bazelbuild/bazel/pull/24703#issuecomment-2665963637
  def testBuildLaunchersWithClangClOnWindows(self):
    if not self.IsWindows():
      return
    self.AddBazelDep('platforms')
    self.AddBazelDep('rules_cc')
    self.ScratchFile(
        'MODULE.bazel',
        [
            'cc_configure = use_extension(',
            '    "@rules_cc//cc:extensions.bzl", "cc_configure_extension")',
            'use_repo(cc_configure, "local_config_cc")',
            # Register all cc toolchains for Windows
            'register_toolchains("@local_config_cc//:all")',
        ],
        mode='a',
    )
    self.ScratchFile(
        'BUILD',
        [
            'platform(',
            '    name = "x64_windows-clang-cl",',
            '    constraint_values = [',
            '        "@platforms//cpu:x86_64",',
            '        "@platforms//os:windows",',
            '        "@bazel_tools//tools/cpp:clang-cl",',
            '    ],',
            ')',
        ],
    )

    exit_code, _, stderr = self.RunBazel([
        'build',
        '--extra_toolchains=@local_config_cc//:cc-toolchain-x64_windows-clang-cl',
        '--extra_execution_platforms=//:x64_windows-clang-cl',
        '--cxxopt=/std:c++17',
        '--host_cxxopt=/std:c++17',
        '@bazel_tools//src/tools/launcher:launcher',
    ])
    self.AssertExitCode(exit_code, 0, stderr)

  def AssertRunfilesManifestContains(self, manifest, entry):
    with open(manifest, 'r') as f:
      for l in f:
        tokens = l.strip().split(' ', 1)
        if len(tokens) == 2 and tokens[0] == entry:
          return
    self.fail('Runfiles manifest "%s" did not contain "%s"' % (manifest, entry))


if __name__ == '__main__':
  absltest.main()
