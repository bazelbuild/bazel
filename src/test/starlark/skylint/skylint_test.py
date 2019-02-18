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

import os.path
import shutil
import subprocess
import tempfile
import unittest

from src.test.starlark.skylint import testenv


class SkylintTest(unittest.TestCase):

  def testGoodFile(self):
    output = subprocess.check_output([
        testenv.SKYLINT_BINARY_PATH,
        os.path.join(testenv.SKYLINT_TESTDATA_PATH, "good.bzl.test")
    ])
    output = output.decode("utf-8")
    self.assertEqual(output, "")

  def testBadFile(self):
    try:
      issues = ""
      subprocess.check_output([
          testenv.SKYLINT_BINARY_PATH,
          os.path.join(testenv.SKYLINT_TESTDATA_PATH, "bad.bzl.test")
      ])
    except subprocess.CalledProcessError as e:
      issues = e.output.decode("utf-8")
    self.assertIn("no module docstring", issues)

  def testNonexistingFile(self):
    try:
      output = ""
      subprocess.check_output(
          [testenv.SKYLINT_BINARY_PATH, "does_not_exist.bzl"],
          stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
      output = e.output.decode("utf-8")
    self.assertEqual("File not found: does_not_exist.bzl\n", output)

  def testDisablingCheck(self):
    output = subprocess.check_output([
        testenv.SKYLINT_BINARY_PATH, "--disable-checks=docstring",
        os.path.join(testenv.SKYLINT_TESTDATA_PATH, "bad.bzl.test")
    ])
    output = output.decode("utf-8")
    self.assertEqual(output, "")

  def testDisablingCategory(self):
    output = subprocess.check_output([
        testenv.SKYLINT_BINARY_PATH,
        "--disable-categories=missing-module-docstring",
        os.path.join(testenv.SKYLINT_TESTDATA_PATH, "bad.bzl.test")
    ])
    output = output.decode("utf-8")
    self.assertEqual(output, "")

  IMPORT_BZL_CONTENTS = """
def foo():
  '''bar

  Deprecated:
    test.'''"""

  def GetOutputOfDependencyTestCase(self, options):
    # Create these dynamically to not interfere with Bazel package structure:
    temp_dir = tempfile.mkdtemp()
    try:
      open(os.path.join(temp_dir, "WORKSPACE"), "a").close()
      open(os.path.join(temp_dir, "BUILD"), "a").close()
      with open(os.path.join(temp_dir, "dependencies.bzl"), "a") as f:
        f.write("'''Docstring'''\nload(':import.bzl', 'foo')\nfoo()")
      with open(os.path.join(temp_dir, "import.bzl"), "a") as f:
        f.write(self.IMPORT_BZL_CONTENTS)
      output = None
      try:
        subprocess.check_output([
            testenv.SKYLINT_BINARY_PATH,
            os.path.join(temp_dir, "dependencies.bzl")
        ] + options)
      except subprocess.CalledProcessError as e:
        output = e.output.decode("utf-8")
      return output
    finally:
      shutil.rmtree(temp_dir)

  def testDependencyAnalysis(self):
    output = self.GetOutputOfDependencyTestCase([])
    self.assertIn("import.bzl) is deprecated: test.", output)

  def testSingleFileModeWorks(self):
    output = self.GetOutputOfDependencyTestCase(["--single-file"])
    self.assertEqual(output, None)


if __name__ == "__main__":
  unittest.main()
