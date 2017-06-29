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
import unittest
from src.test.py.bazel import test_base


class LauncherScriptTest(test_base.TestBase):

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
      self.AssertRunfilesManifestContains(
          os.path.join(bazel_bin, 'foo', 'foo.runfiles', 'MANIFEST'),
          '__main__/bar/bar.txt')
    else:
      self.assertTrue(
          os.path.islink(
              os.path.join(bazel_bin, 'foo/foo.runfiles/__main__/bar/bar.txt')))

    exit_code, stdout, stderr = self.RunProgram([main_binary])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertEqual(stdout[0], 'hello java')

  def AssertRunfilesManifestContains(self, manifest, entry):
    with open(manifest, 'r') as f:
      for l in f:
        tokens = l.strip().split(' ', 1)
        if len(tokens) == 2 and tokens[0] == entry:
          return
    self.fail('Runfiles manifest "%s" did not contain "%s"' % (manifest, entry))


if __name__ == '__main__':
  unittest.main()
