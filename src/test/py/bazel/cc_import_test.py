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
import shutil
import stat
from absl.testing import absltest
from src.test.py.bazel import test_base


class CcImportTest(test_base.TestBase):

  def createProjectFiles(self,
                         alwayslink=0,
                         system_provided=0,
                         linkstatic=1,
                         provide_header=True):

    # We use the outputs of cc_binary and cc_library as precompiled
    # libraries for cc_import
    self.ScratchFile(
        'lib/BUILD',
        [
            'package(default_visibility = ["//visibility:public"])',
            '',
            'cc_binary(',
            '  name = "libA.so",',
            '  srcs = ["a.cc"],',
            '  linkshared = 1,',
            ')',
            '',
            'filegroup(',
            '  name = "libA_ifso",',
            '  srcs = [":libA.so"],',
            '  output_group = "interface_library",',
            ')',
            '',
            'cc_library(',
            '  name = "libA",',
            '  srcs = ["a.cc", "a_al.cc"],',
            ')',
            '',
            'filegroup(',
            '  name = "libA_archive",',
            '  srcs = [":libA"],',
            '  output_group = "archive",',
            ')',
            '',
            'cc_import(',
            '  name = "A",',
            '  static_library = "//lib:libA_archive",',
            '  shared_library = "//lib:libA.so",'
            if not system_provided else '',
            # On Windows, we always need the interface library
            '  interface_library = "//lib:libA_ifso",'
            if self.IsWindows() else (
                # On Unix, we use .so file as interface library
                # if system_provided is true
                '  interface_library = "//lib:libA.so",'
                if system_provided else ''),
            '  hdrs = ["a.h"],' if provide_header else '',
            '  alwayslink = %s,' % str(alwayslink),
            '  system_provided = %s,' % str(system_provided),
            ')',
        ])

    self.ScratchFile('lib/a.cc', [
        '#include <stdio.h>',
        '',
        '#ifdef _WIN32',
        '  #define DLLEXPORT __declspec(dllexport)',
        '#else',
        '  #define DLLEXPORT',
        '#endif',
        '',
        'DLLEXPORT void HelloWorld() {',
        '  printf("HelloWorld\\n");',
        '}',
    ])

    # For testing alwayslink=1
    self.ScratchFile('lib/a_al.cc', [
        'extern int global_variable;',
        'int init() {',
        '    ++global_variable;',
        '    return global_variable;',
        '}',
        'int x = init();',
        'int y = init();',
    ])

    self.ScratchFile('lib/a.h', [
        'void HelloWorld();',
    ])

    self.ScratchFile('main/BUILD', [
        'cc_binary(',
        '  name = "B",',
        '  srcs = ["b.cc"],',
        '  deps = ["//lib:A",],',
        '  linkstatic = %s,' % str(linkstatic),
        ')',
    ])

    self.ScratchFile('main/b.cc', [
        '#include <stdio.h>',
        '#include "lib/a.h"',
        'int global_variable = 0;',
        'int main() {',
        '  HelloWorld();',
        '  printf("global : %d\\n", global_variable);',
        '  return 0;',
        '}',
    ])

  def getBazelInfo(self, info_key):
    _, stdout, _ = self.RunBazel(['info', info_key])
    return stdout[0]

  def testLinkStaticLibrary(self):
    self.createProjectFiles(alwayslink=0, linkstatic=1)
    bazel_bin = self.getBazelInfo('bazel-bin')
    suffix = '.exe' if self.IsWindows() else ''

    self.RunBazel(['build', '//main:B'])

    b_bin = os.path.join(bazel_bin, 'main/B' + suffix)
    self.assertTrue(os.path.exists(b_bin))
    _, stdout, _ = self.RunProgram([b_bin])
    self.assertEqual(stdout[0], 'HelloWorld')
    self.assertEqual(stdout[1], 'global : 0')

  def testAlwayslinkStaticLibrary(self):
    self.createProjectFiles(alwayslink=1, linkstatic=1)
    bazel_bin = self.getBazelInfo('bazel-bin')
    suffix = '.exe' if self.IsWindows() else ''

    self.RunBazel(['build', '//main:B'])

    b_bin = os.path.join(bazel_bin, 'main/B' + suffix)
    self.assertTrue(os.path.exists(b_bin))
    _, stdout, _ = self.RunProgram([b_bin])
    self.assertEqual(stdout[0], 'HelloWorld')
    self.assertEqual(stdout[1], 'global : 2')

  def testLinkSharedLibrary(self):
    self.createProjectFiles(linkstatic=0)
    bazel_bin = self.getBazelInfo('bazel-bin')
    suffix = '.exe' if self.IsWindows() else ''

    self.RunBazel(['build', '//main:B'])

    b_bin = os.path.join(bazel_bin, 'main/B' + suffix)
    self.assertTrue(os.path.exists(b_bin))
    if self.IsWindows():
      self.assertTrue(os.path.exists(os.path.join(bazel_bin, 'main/libA.so')))
    _, stdout, _ = self.RunProgram([b_bin])
    self.assertEqual(stdout[0], 'HelloWorld')

  def testSystemProvidedSharedLibraryOnWinodws(self):
    if not self.IsWindows():
      return
    self.createProjectFiles(system_provided=1, linkstatic=0)
    bazel_bin = self.getBazelInfo('bazel-bin')

    self.RunBazel(['build', '//main:B'])

    b_bin = os.path.join(bazel_bin, 'main/B.exe')
    exit_code, _, _ = self.RunProgram([b_bin], allow_failure=True)
    # Should fail because missing libA.so
    self.assertFalse(exit_code == 0)

    # Let's build libA.so and add it into PATH
    self.RunBazel(['build', '//lib:libA.so'])

    _, stdout, _ = self.RunProgram(
        [b_bin], env_add={'PATH': str(os.path.join(bazel_bin, 'lib'))}
    )
    self.assertEqual(stdout[0], 'HelloWorld')

  def testSystemProvidedSharedLibraryOnUnix(self):
    if not self.IsLinux():
      return
    self.createProjectFiles(system_provided=1, linkstatic=0)
    bazel_bin = self.getBazelInfo('bazel-bin')

    self.RunBazel(['build', '//main:B'])

    b_bin = os.path.join(bazel_bin, 'main/B')
    tmp_dir = self.ScratchDir('temp_dir_for_run_b_bin')
    b_bin_tmp = os.path.join(tmp_dir, 'B')
    # Copy the binary to a temp directory to make sure it cannot find
    # libA.so
    shutil.copyfile(b_bin, b_bin_tmp)
    os.chmod(b_bin_tmp, stat.S_IRWXU)
    exit_code, _, _ = self.RunProgram([b_bin_tmp], allow_failure=True)
    # Should fail because missing libA.so
    self.assertFalse(exit_code == 0)

    # Let's build libA.so and add it into PATH
    self.RunBazel(['build', '//lib:libA.so'])

    _, stdout, _ = self.RunProgram(
        [b_bin_tmp],
        env_add={
            # For Linux
            'LD_LIBRARY_PATH': str(os.path.join(bazel_bin, 'lib')),
            # For Mac
            'DYLD_LIBRARY_PATH': str(os.path.join(bazel_bin, 'lib')),
        },
    )
    self.assertEqual(stdout[0], 'HelloWorld')

  def testCcImportHeaderCheck(self):
    self.createProjectFiles(provide_header=False)
    # Build should fail, because lib/a.h is not declared in BUILD file, disable
    # sandbox so that bazel produces same error across different platforms.
    exit_code, _, stderr = self.RunBazel(
        ['build', '//main:B', '--spawn_strategy=standalone'], allow_failure=True
    )
    self.AssertExitCode(exit_code, 1, stderr)
    self.assertIn('this rule is missing dependency declarations for the'
                  ' following files included by \'main/b.cc\':',
                  ''.join(stderr))


if __name__ == '__main__':
  absltest.main()
