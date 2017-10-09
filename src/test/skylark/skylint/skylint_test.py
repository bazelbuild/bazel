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
import subprocess
import unittest

from src.test.skylark.skylint import testenv


class SkylintTest(unittest.TestCase):

  def testGoodFile(self):
    output = subprocess.check_output([
        testenv.SKYLINT_BINARY_PATH,
        os.path.join(testenv.SKYLINT_TESTDATA_PATH, "good.bzl.test")
    ])
    self.assertEqual(output, "")

  def testBadFile(self):
    try:
      issues = ""
      subprocess.check_output([
          testenv.SKYLINT_BINARY_PATH,
          os.path.join(testenv.SKYLINT_TESTDATA_PATH, "bad.bzl.test")
      ])
    except subprocess.CalledProcessError as e:
      issues = e.output
    self.assertIn("no module docstring", issues)

  def testDisablingChecker(self):
    output = subprocess.check_output([
        testenv.SKYLINT_BINARY_PATH, "--disable=docstring",
        os.path.join(testenv.SKYLINT_TESTDATA_PATH, "bad.bzl.test")
    ])
    self.assertEqual(output, "")


if __name__ == "__main__":
  unittest.main()
