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


class BazelWindowsCppTest(test_base.TestBase):

  def createProjectFiles(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('BUILD', [
        'package(',
        '  default_visibility = ["//visibility:public"],',
        '  features=["windows_export_all_symbols"]',
        ')',
        '',
        'cc_library(',
        '  name = "A",',
        '  srcs = ["a.cc"],',
        '  hdrs = ["a.h"],',
        '  copts = ["/DCOMPILING_A_DLL"],',
        '  features = ["no_windows_export_all_symbols"],',
        ')',
        '',
        'cc_library(',
        '  name = "B",',
        '  srcs = ["b.cc"],',
        '  hdrs = ["b.h"],',
        '  deps = [":A"],',
        '  copts = ["/DNO_DLLEXPORT"],',
        ')',
        '',
        'cc_binary(',
        '  name = "C",',
        '  srcs = ["c.cc"],',
        '  deps = [":A", ":B" ],',
        '  linkstatic = 0,',
        ')',
    ])

    self.ScratchFile('a.cc', [
        '#include <stdio.h>',
        '#include "a.h"',
        'int a = 0;',
        'void hello_A() {',
        '  a++;',
        '  printf("Hello A, %d\\n", a);',
        '}',
    ])

    self.ScratchFile('b.cc', [
        '#include <stdio.h>',
        '#include "a.h"',
        '#include "b.h"',
        'void hello_B() {',
        '  hello_A();',
        '  printf("Hello B\\n");',
        '}',
    ])
    header_temp = [
        '#ifndef %{name}_H',
        '#define %{name}_H',
        '',
        '#if NO_DLLEXPORT',
        '  #define DLLEXPORT',
        '#elif COMPILING_%{name}_DLL',
        '  #define DLLEXPORT __declspec(dllexport)',
        '#else',
        '  #define DLLEXPORT __declspec(dllimport)',
        '#endif',
        '',
        'DLLEXPORT void hello_%{name}();',
        '',
        '#endif',
    ]
    self.ScratchFile('a.h',
                     [line.replace('%{name}', 'A') for line in header_temp])
    self.ScratchFile('b.h',
                     [line.replace('%{name}', 'B') for line in header_temp])

    c_cc_content = [
        '#include <stdio.h>',
        '#include "a.h"',
        '#include "b.h"',
        '',
        'void hello_C() {',
        '  hello_A();',
        '  hello_B();',
        '  printf("Hello C\\n");',
        '}',
        '',
        'int main() {',
        '  hello_C();',
        '  return 0;',
        '}',
    ]

    self.ScratchFile('c.cc', c_cc_content)

    self.ScratchFile('lib/BUILD', [
        'cc_library(',
        '  name = "A",',
        '  srcs = ["dummy.cc"],',
        '  features = ["windows_export_all_symbols"],',
        '  visibility = ["//visibility:public"],',
        ')',
    ])
    self.ScratchFile('lib/dummy.cc', ['void dummy() {}'])

    self.ScratchFile('main/main.cc', c_cc_content)

  def getBazelInfo(self, info_key):
    exit_code, stdout, stderr = self.RunBazel(['info', info_key])
    self.AssertExitCode(exit_code, 0, stderr)
    return stdout[0]

  def testBuildDynamicLibraryWithUserExportedSymbol(self):
    self.createProjectFiles()
    bazel_bin = self.getBazelInfo('bazel-bin')

    # //:A export symbols by itself using __declspec(dllexport), so it doesn't
    # need Bazel to export symbols using DEF file.
    exit_code, _, stderr = self.RunBazel(
        ['build', '//:A', '--output_groups=dynamic_library'])
    self.AssertExitCode(exit_code, 0, stderr)

    # TODO(pcloudy): change suffixes to .lib and .dll after making DLL
    # extensions correct on
    # Windows.
    import_library = os.path.join(bazel_bin, 'libA.ifso')
    shared_library = os.path.join(bazel_bin, 'libA.so')
    def_file = os.path.join(bazel_bin, 'A.def')
    self.assertTrue(os.path.exists(import_library))
    self.assertTrue(os.path.exists(shared_library))
    # DEF file shouldn't be generated for //:A
    self.assertFalse(os.path.exists(def_file))

  def testBuildDynamicLibraryWithExportSymbolFeature(self):
    self.createProjectFiles()
    bazel_bin = self.getBazelInfo('bazel-bin')

    # //:B doesn't export symbols by itself, so it need Bazel to export symbols
    # using DEF file.
    exit_code, _, stderr = self.RunBazel(
        ['build', '//:B', '--output_groups=dynamic_library'])
    self.AssertExitCode(exit_code, 0, stderr)

    # TODO(pcloudy): change suffixes to .lib and .dll after making DLL
    # extensions correct on
    # Windows.
    import_library = os.path.join(bazel_bin, 'libB.ifso')
    shared_library = os.path.join(bazel_bin, 'libB.so')
    def_file = os.path.join(bazel_bin, 'B.def')
    self.assertTrue(os.path.exists(import_library))
    self.assertTrue(os.path.exists(shared_library))
    # DEF file should be generated for //:B
    self.assertTrue(os.path.exists(def_file))

    # Test build //:B if windows_export_all_symbols feature is disabled by
    # no_windows_export_all_symbols.
    exit_code, _, stderr = self.RunBazel([
        'build', '//:B', '--output_groups=dynamic_library',
        '--features=no_windows_export_all_symbols'
    ])
    self.AssertExitCode(exit_code, 1, stderr)
    self.assertIn('output \'libB.ifso\' was not created', ''.join(stderr))

  def testBuildCcBinaryWithDependenciesDynamicallyLinked(self):
    self.createProjectFiles()
    bazel_bin = self.getBazelInfo('bazel-bin')

    # Since linkstatic=0 is specified for //:C, it's dependencies should be
    # dynamically linked.
    exit_code, _, stderr = self.RunBazel(['build', '//:C'])
    self.AssertExitCode(exit_code, 0, stderr)

    # TODO(pcloudy): change suffixes to .lib and .dll after making DLL
    # extensions correct on
    # Windows.
    # a_import_library
    self.assertTrue(os.path.exists(os.path.join(bazel_bin, 'libA.ifso')))
    # a_shared_library
    self.assertTrue(os.path.exists(os.path.join(bazel_bin, 'libA.so')))
    # a_def_file
    self.assertFalse(os.path.exists(os.path.join(bazel_bin, 'A.def')))
    # b_import_library
    self.assertTrue(os.path.exists(os.path.join(bazel_bin, 'libB.ifso')))
    # b_shared_library
    self.assertTrue(os.path.exists(os.path.join(bazel_bin, 'libB.so')))
    # b_def_file
    self.assertTrue(os.path.exists(os.path.join(bazel_bin, 'B.def')))
    # c_exe
    self.assertTrue(os.path.exists(os.path.join(bazel_bin, 'C.exe')))

  def testBuildCcBinaryFromDifferentPackage(self):
    self.createProjectFiles()
    self.ScratchFile('main/BUILD', [
        'cc_binary(',
        '  name = "main",',
        '  srcs = ["main.cc"],',
        '  deps = ["//:B"],',
        '  linkstatic = 0,'
        ')',
    ])
    bazel_bin = self.getBazelInfo('bazel-bin')

    exit_code, _, stderr = self.RunBazel(['build', '//main:main'])
    self.AssertExitCode(exit_code, 0, stderr)

    # Test if libA.so and libB.so are copied to the directory of main.exe
    main_bin = os.path.join(bazel_bin, 'main/main.exe')
    self.assertTrue(os.path.exists(main_bin))
    self.assertTrue(os.path.exists(os.path.join(bazel_bin, 'main/libA.so')))
    self.assertTrue(os.path.exists(os.path.join(bazel_bin, 'main/libB.so')))

    # Run the binary to see if it runs successfully
    exit_code, stdout, stderr = self.RunProgram([main_bin])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertEqual(['Hello A, 1', 'Hello A, 2', 'Hello B', 'Hello C'], stdout)

  def testBuildCcBinaryDependsOnConflictDLLs(self):
    self.createProjectFiles()
    self.ScratchFile(
        'main/BUILD',
        [
            'cc_binary(',
            '  name = "main",',
            '  srcs = ["main.cc"],',
            '  deps = ["//:B", "//lib:A"],',  # Transitively depends on //:A
            '  linkstatic = 0,'
            ')',
        ])

    # //main:main depends on both //lib:A and //:A,
    # their dlls are both called libA.so,
    # so there should be a conflict error
    exit_code, _, stderr = self.RunBazel(['build', '//main:main'])
    self.AssertExitCode(exit_code, 1, stderr)
    self.assertIn(
        'ERROR: file \'main/libA.so\' is generated by these conflicting '
        'actions:', ''.join(stderr))

  def testBuildDifferentCcBinariesDependOnConflictDLLs(self):
    self.createProjectFiles()
    self.ScratchFile(
        'main/BUILD',
        [
            'cc_binary(',
            '  name = "main",',
            '  srcs = ["main.cc"],',
            '  deps = ["//:B"],',  # Transitively depends on //:A
            '  linkstatic = 0,'
            ')',
            '',
            'cc_binary(',
            '  name = "other_main",',
            '  srcs = ["other_main.cc"],',
            '  deps = ["//lib:A"],',
            '  linkstatic = 0,'
            ')',
        ])
    self.ScratchFile('main/other_main.cc', ['int main() {return 0;}'])

    # Building //main:main should succeed
    exit_code, _, stderr = self.RunBazel(['build', '//main:main'])
    self.AssertExitCode(exit_code, 0, stderr)

    # Building //main:other_main after //main:main should fail
    exit_code, _, stderr = self.RunBazel(['build', '//main:other_main'])
    self.AssertExitCode(exit_code, 1, stderr)
    self.assertIn(
        'ERROR: file \'main/libA.so\' is generated by these conflicting '
        'actions:', ''.join(stderr))

  def testDynamicLinkingMSVCRT(self):
    self.createProjectFiles()

    # By default, it should link to msvcrt dynamically.
    exit_code, _, stderr = self.RunBazel(
        ['build', '//:A', '--output_groups=dynamic_library', '-s'])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn('/MD', ''.join(stderr))
    self.assertIn('/DEFAULTLIB:msvcrt.lib', ''.join(stderr))
    self.assertNotIn('/MT', ''.join(stderr))
    self.assertNotIn('/DEFAULTLIB:libcmt.lib', ''.join(stderr))

    # Test build in debug mode.
    exit_code, _, stderr = self.RunBazel(
        ['build', '-c', 'dbg', '//:A', '--output_groups=dynamic_library', '-s'])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn('/MDd', ''.join(stderr))
    self.assertIn('/DEFAULTLIB:msvcrtd.lib', ''.join(stderr))
    self.assertNotIn('/MTd', ''.join(stderr))
    self.assertNotIn('/DEFAULTLIB:libcmtd.lib', ''.join(stderr))

  def testStaticLinkingMSVCRT(self):
    self.createProjectFiles()

    # With static_link_msvcrt feature, it should link to msvcrt statically.
    exit_code, _, stderr = self.RunBazel([
        'build', '//:A', '--output_groups=dynamic_library',
        '--features=static_link_msvcrt', '-s'
    ])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertNotIn('/MD', ''.join(stderr))
    self.assertNotIn('/DEFAULTLIB:msvcrt.lib', ''.join(stderr))
    self.assertIn('/MT', ''.join(stderr))
    self.assertIn('/DEFAULTLIB:libcmt.lib', ''.join(stderr))

    # Test build in debug mode.
    exit_code, _, stderr = self.RunBazel([
        'build', '-c', 'dbg', '//:A', '--output_groups=dynamic_library',
        '--features=static_link_msvcrt', '-s'
    ])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertNotIn('/MDd', ''.join(stderr))
    self.assertNotIn('/DEFAULTLIB:msvcrtd.lib', ''.join(stderr))
    self.assertIn('/MTd', ''.join(stderr))
    self.assertIn('/DEFAULTLIB:libcmtd.lib', ''.join(stderr))

  def testBuildSharedLibraryFromCcBinaryWithStaticLink(self):
    self.createProjectFiles()
    self.ScratchFile(
        'main/BUILD',
        [
            'cc_binary(',
            '  name = "main.so",',
            '  srcs = ["main.cc"],',
            '  deps = ["//:B"],',  # Transitively depends on //:A
            '  linkstatic = 1,'
            '  linkshared = 1,'
            '  features=["windows_export_all_symbols"]',
            ')',
        ])
    bazel_bin = self.getBazelInfo('bazel-bin')

    exit_code, _, stderr = self.RunBazel([
        'build', '//main:main.so', '--output_groups=default,interface_library'
    ])
    self.AssertExitCode(exit_code, 0, stderr)

    main_library = os.path.join(bazel_bin, 'main/main.so')
    main_interface = os.path.join(bazel_bin, 'main/main.ifso')
    def_file = os.path.join(bazel_bin, 'main/main.so.def')
    self.assertTrue(os.path.exists(main_library))
    self.assertTrue(os.path.exists(main_interface))
    self.assertTrue(os.path.exists(def_file))
    # libA.so and libB.so should not be copied.
    self.assertFalse(os.path.exists(os.path.join(bazel_bin, 'main/libA.so')))
    self.assertFalse(os.path.exists(os.path.join(bazel_bin, 'main/libB.so')))
    self.AssertFileContentContains(def_file, 'hello_A')
    self.AssertFileContentContains(def_file, 'hello_B')
    self.AssertFileContentContains(def_file, 'hello_C')

  def testBuildSharedLibraryFromCcBinaryWithDynamicLink(self):
    self.createProjectFiles()
    self.ScratchFile(
        'main/BUILD',
        [
            'cc_binary(',
            '  name = "main.so",',
            '  srcs = ["main.cc"],',
            '  deps = ["//:B"],',  # Transitively depends on //:A
            '  linkstatic = 0,'
            '  linkshared = 1,'
            '  features=["windows_export_all_symbols"]',
            ')',
        ])
    bazel_bin = self.getBazelInfo('bazel-bin')

    exit_code, _, stderr = self.RunBazel([
        'build', '//main:main.so', '--output_groups=default,interface_library'
    ])
    self.AssertExitCode(exit_code, 0, stderr)

    main_library = os.path.join(bazel_bin, 'main/main.so')
    main_interface = os.path.join(bazel_bin, 'main/main.ifso')
    def_file = os.path.join(bazel_bin, 'main/main.so.def')
    self.assertTrue(os.path.exists(main_library))
    self.assertTrue(os.path.exists(main_interface))
    self.assertTrue(os.path.exists(def_file))
    # libA.so and libB.so should be copied.
    self.assertTrue(os.path.exists(os.path.join(bazel_bin, 'main/libA.so')))
    self.assertTrue(os.path.exists(os.path.join(bazel_bin, 'main/libB.so')))
    # hello_A and hello_B should not be exported.
    self.AssertFileContentNotContains(def_file, 'hello_A')
    self.AssertFileContentNotContains(def_file, 'hello_B')
    self.AssertFileContentContains(def_file, 'hello_C')

  def testGetDefFileOfSharedLibraryFromCcBinary(self):
    self.createProjectFiles()
    self.ScratchFile(
        'main/BUILD',
        [
            'cc_binary(',
            '  name = "main.so",',
            '  srcs = ["main.cc"],',
            '  deps = ["//:B"],',  # Transitively depends on //:A
            '  linkstatic = 1,'
            '  linkshared = 1,'
            ')',
        ])
    bazel_bin = self.getBazelInfo('bazel-bin')

    exit_code, _, stderr = self.RunBazel(
        ['build', '//main:main.so', '--output_groups=def_file'])
    self.AssertExitCode(exit_code, 0, stderr)

    # Although windows_export_all_symbols is not specified for this target,
    # we should still be able to get the DEF file by def_file output group.
    def_file = os.path.join(bazel_bin, 'main/main.so.def')
    self.assertTrue(os.path.exists(def_file))
    self.AssertFileContentContains(def_file, 'hello_A')
    self.AssertFileContentContains(def_file, 'hello_B')
    self.AssertFileContentContains(def_file, 'hello_C')

  def AssertFileContentContains(self, file_path, entry):
    f = open(file_path, 'r')
    if entry not in f.read():
      self.fail('File "%s" does not contain "%s"' % (file_path, entry))

  def AssertFileContentNotContains(self, file_path, entry):
    f = open(file_path, 'r')
    if entry in f.read():
      self.fail('File "%s" does contain "%s"' % (file_path, entry))

  def testWinDefFileAttribute(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('lib.cc', ['void hello() {}'])
    self.ScratchFile('my_lib.def', [
        'LIBRARY liblib.so',
        'EXPORTS',
        '        ?hello@@YAXXZ',
    ])
    self.ScratchFile('BUILD', [
        'cc_library(',
        '  name = "lib",',
        '  srcs = ["lib.cc"],',
        '  win_def_file = "my_lib.def",',
        ')',
        '',
        'cc_binary(',
        '  name = "lib.so",',
        '  srcs = ["lib.cc"],',
        '  win_def_file = "my_lib.def",',
        '  linkshared = 1,',
        ')',
    ])

    # Test exporting symbols using custom DEF file in cc_library.
    # Auto-generating DEF file should be disbaled when custom DEF file specified
    exit_code, _, stderr = self.RunBazel([
        'build', '//:lib', '-s', '--output_groups=dynamic_library',
        '--features=windows_export_all_symbols'
    ])

    bazel_bin = self.getBazelInfo('bazel-bin')
    lib_ifso = os.path.join(bazel_bin, 'liblib.ifso')
    lib_def = os.path.join(bazel_bin, 'lib.def')
    self.assertTrue(os.path.exists(lib_ifso))
    self.assertFalse(os.path.exists(lib_def))

    # Test specifying DEF file in cc_binary
    exit_code, _, stderr = self.RunBazel(['build', '//:lib.so', '-s'])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertIn('/DEF:my_lib.def', ''.join(stderr))

  def testCcImportRule(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('BUILD', [
        'cc_import(',
        '  name = "a_import",',
        '  static_library = "A.a",',
        '  shared_library = "A.so",',
        '  interface_library = "A.ifso",',
        '  hdrs = ["a.h"],',
        '  alwayslink = 1,',
        ')',
    ])
    exit_code, _, stderr = self.RunBazel([
        'build', '//:a_import',
    ])
    self.AssertExitCode(exit_code, 0, stderr)

if __name__ == '__main__':
  unittest.main()
