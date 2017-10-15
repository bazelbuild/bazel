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

class DEFParserTest(test_base.TestBase):

  def createAndBuildProjectFiles(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('BUILD', ['cc_library(name="hello", srcs=["x.cc"])'])
    self.ScratchFile('x.cc', [
        '#include <stdio.h>',
        'int hello_data;',
        'void hello_world() {',
        '  printf("hello world\\n");',
        '}',
    ])
    exit_code, _, stderr = self.RunBazel(['build', '//:hello'])
    self.AssertExitCode(exit_code, 0, stderr)

  def testParseDefFileFromObjectFile(self):
    # Skip this test on non-Windows platforms
    if not self.IsWindows():
      return
    self.createAndBuildProjectFiles()

    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-bin'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    objfile = os.path.join(bazel_bin, '_objs', 'hello', 'x.o')
    self.assertTrue(os.path.isfile(objfile))
    output_def = self.Path('x.def');
    self.RunProgram([self.Rlocation('io_bazel/third_party/def_parser/def_parser.exe'), output_def, 'my_x.dll', objfile])
    self.assertTrue(os.path.isfile(output_def))

    with open(output_def, 'r') as def_file:
      def_content = def_file.read()
      self.assertIn('LIBRARY my_x.dll', def_content)
      self.assertIn('hello_data', def_content)
      self.assertIn('hello_world', def_content)

  def testParseDefFileFromObjectFileWithParamFile(self):
    # Skip this test on non-Windows platforms
    if not self.IsWindows():
      return
    self.createAndBuildProjectFiles()

    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-bin'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    objfile = os.path.join(bazel_bin, '_objs', 'hello', 'x.o')
    self.assertTrue(os.path.isfile(objfile))
    objfilelist = self.ScratchFile('objfilelist', [objfile])

    output_def = self.Path('x.def');
    self.RunProgram([self.Rlocation('io_bazel/third_party/def_parser/def_parser.exe'), output_def, 'my_x.dll', '@' + objfilelist])
    self.assertTrue(os.path.isfile(output_def))

    with open(output_def, 'r') as def_file:
      def_content = def_file.read()
      self.assertIn('LIBRARY my_x.dll', def_content)
      self.assertIn('hello_data', def_content)
      self.assertIn('hello_world', def_content)

  def testParseDefFileFromAnotherDefFile(self):
    # Skip this test on non-Windows platforms
    if not self.IsWindows():
      return

    self.createAndBuildProjectFiles()

    exit_code, stdout, stderr = self.RunBazel(['info', 'bazel-bin'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    objfile = os.path.join(bazel_bin, '_objs', 'hello', 'x.o')
    self.assertTrue(os.path.isfile(objfile))
    output_def = self.Path('x.def');
    self.RunProgram([self.Rlocation('io_bazel/third_party/def_parser/def_parser.exe'), output_def, 'my_x.dll', objfile])
    self.assertTrue(os.path.isfile(output_def))

    new_output_def = self.Path('new_x.def');
    self.RunProgram([self.Rlocation('io_bazel/third_party/def_parser/def_parser.exe'), new_output_def, 'my_x.dll', output_def])
    self.assertTrue(os.path.isfile(new_output_def))

    with open(new_output_def, 'r') as def_file:
      def_content = def_file.read()
      self.assertIn('LIBRARY my_x.dll', def_content)
      self.assertIn('hello_data', def_content)
      self.assertIn('hello_world', def_content)

if __name__ == '__main__':
  unittest.main()
