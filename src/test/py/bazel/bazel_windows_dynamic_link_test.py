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


class BazelWindowsDynamicLinkTest(test_base.TestBase):

  def createProjectFiles(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('BUILD', [
        'package(',
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

    self.ScratchFile('c.cc', [
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
    ])

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
        ['build', '//:A', '--output_groups=cc_dynamic_library'])
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
        ['build', '//:B', '--output_groups=cc_dynamic_library'])
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
        'build', '//:B', '--output_groups=cc_dynamic_library',
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


if __name__ == '__main__':
  unittest.main()
