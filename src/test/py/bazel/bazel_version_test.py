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


class BazelVersionTest(test_base.TestBase):

  def testStartupVersionOption(self):
    exit_code, stdout, stderr = self.RunBazel(["--version"])
    self.AssertExitCode(exit_code, 0, stderr)
    version_info = stdout[0]
    self.assertEqual(version_info, "bazel no_version")

    # build //... -s should be ignored since --version is passed
    exit_code, stdout, stderr = self.RunBazel(["--version", "build", "//...", "-s"])
    self.AssertExitCode(exit_code, 0, stderr)
    version_info = stdout[0]
    self.assertEqual(version_info, "bazel no_version")

  def testStartupVersionOptionWithTrustedInstallBase(self):
    install_base = self.ScratchDir("my_install_base")
    self.ScratchFile("my_install_base/build-label.txt", ["x.x.x"])

    exit_code, stdout, stderr = self.RunBazel([
        "--version", "--install_base=%s" % install_base, "--trust_install_base"])
    self.AssertExitCode(exit_code, 0, stderr)
    version_info = stdout[0]
    self.assertEqual(version_info, "bazel x.x.x")


if __name__ == "__main__":
  unittest.main()
