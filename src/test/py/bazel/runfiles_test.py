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
import six
from src.test.py.bazel import test_base


class RunfilesTest(test_base.TestBase):

  def testAttemptToBuildRunfilesOnWindows(self):
    if not self.IsWindows():
      self.skipTest("only applicable to Windows")
    self.ScratchFile("WORKSPACE")
    exit_code, _, stderr = self.RunBazel(
        ["build", "--experimental_enable_runfiles"])
    self.assertNotEqual(exit_code, 0)
    self.assertIn("building runfiles is not supported on Windows",
                  "\n".join(stderr))

  def testJavaRunfilesLibraryInBazelToolsRepo(self):
    for s, t in [
        ("WORKSPACE.mock", "WORKSPACE"),
        ("foo/BUILD.mock", "foo/BUILD"),
        ("foo/Foo.java", "foo/Foo.java"),
        ("foo/datadep/hello.txt", "foo/datadep/hello.txt"),
    ]:
      self.CopyFile(
          self.Rlocation(
              "io_bazel/src/test/py/bazel/testdata/runfiles_test/" + s), t)

    exit_code, stdout, stderr = self.RunBazel(["info", "bazel-bin"])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    exit_code, _, stderr = self.RunBazel(["build", "//foo:runfiles-java"])
    self.AssertExitCode(exit_code, 0, stderr)

    if test_base.TestBase.IsWindows():
      bin_path = os.path.join(bazel_bin, "foo/runfiles-java.exe")
    else:
      bin_path = os.path.join(bazel_bin, "foo/runfiles-java")

    self.assertTrue(os.path.exists(bin_path))

    exit_code, stdout, stderr = self.RunProgram(
        [bin_path], env_add={"TEST_SRCDIR": "__ignore_me__"})
    self.AssertExitCode(exit_code, 0, stderr)
    if len(stdout) != 2:
      self.fail("stdout: %s" % stdout)
    self.assertEqual(stdout[0], "Hello Java Foo!")
    six.assertRegex(self, stdout[1], "^rloc=.*/foo/datadep/hello.txt")
    self.assertNotIn("__ignore_me__", stdout[1])
    with open(stdout[1].split("=", 1)[1], "r") as f:
      lines = [l.strip() for l in f.readlines()]
    if len(lines) != 1:
      self.fail("lines: %s" % lines)
    self.assertEqual(lines[0], "world")

  def testPythonRunfilesLibraryInBazelToolsRepo(self):
    for s, t in [
        ("WORKSPACE.mock", "WORKSPACE"),
        ("foo/BUILD.mock", "foo/BUILD"),
        ("foo/runfiles.py", "foo/runfiles.py"),
        ("foo/datadep/hello.txt", "foo/datadep/hello.txt"),
        ("bar/BUILD.mock", "bar/BUILD"),
        ("bar/bar.py", "bar/bar.py"),
        ("bar/bar-py-data.txt", "bar/bar-py-data.txt"),
    ]:
      self.CopyFile(
          self.Rlocation(
              "io_bazel/src/test/py/bazel/testdata/runfiles_test/" + s), t)

    exit_code, stdout, stderr = self.RunBazel(["info", "bazel-bin"])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    exit_code, _, stderr = self.RunBazel(["build", "//foo:runfiles-py"])
    self.AssertExitCode(exit_code, 0, stderr)

    if test_base.TestBase.IsWindows():
      bin_path = os.path.join(bazel_bin, "foo/runfiles-py.exe")
    else:
      bin_path = os.path.join(bazel_bin, "foo/runfiles-py")

    self.assertTrue(os.path.exists(bin_path))

    exit_code, stdout, stderr = self.RunProgram(
        [bin_path], env_add={"TEST_SRCDIR": "__ignore_me__"})
    self.AssertExitCode(exit_code, 0, stderr)
    if len(stdout) < 4:
      self.fail("stdout: %s" % stdout)
    self.assertEqual(stdout[0], "Hello Python Foo!")
    six.assertRegex(self, stdout[1], "^rloc=.*/foo/datadep/hello.txt")
    self.assertNotIn("__ignore_me__", stdout[1])
    self.assertEqual(stdout[2], "Hello Python Bar!")
    six.assertRegex(self, stdout[3], "^rloc=.*/bar/bar-py-data.txt")
    self.assertNotIn("__ignore_me__", stdout[3])

    with open(stdout[1].split("=", 1)[1], "r") as f:
      lines = [l.strip() for l in f.readlines()]
    if len(lines) != 1:
      self.fail("lines: %s" % lines)
    self.assertEqual(lines[0], "world")

    with open(stdout[3].split("=", 1)[1], "r") as f:
      lines = [l.strip() for l in f.readlines()]
    if len(lines) != 1:
      self.fail("lines: %s" % lines)
    self.assertEqual(lines[0], "data for bar.py")

  def testPythonRunfilesLibraryFindsRunfilesWithoutEnvvars(self):
    for s, t in [
        ("WORKSPACE.mock", "WORKSPACE"),
        ("bar/BUILD.mock", "bar/BUILD"),
        ("bar/bar.py", "bar/bar.py"),
        ("bar/bar-py-data.txt", "bar/bar-py-data.txt"),
    ]:
      self.CopyFile(
          self.Rlocation(
              "io_bazel/src/test/py/bazel/testdata/runfiles_test/" + s), t)

    exit_code, stdout, stderr = self.RunBazel(["info", "bazel-bin"])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    exit_code, _, stderr = self.RunBazel(["build", "//bar:all"])
    self.AssertExitCode(exit_code, 0, stderr)

    if test_base.TestBase.IsWindows():
      bin_path = os.path.join(bazel_bin, "bar/bar-py.exe")
    else:
      bin_path = os.path.join(bazel_bin, "bar/bar-py")

    self.assertTrue(os.path.exists(bin_path))

    exit_code, stdout, stderr = self.RunProgram(
        [bin_path],
        env_remove=set([
            "RUNFILES_MANIFEST_FILE",
            "RUNFILES_MANIFEST_ONLY",
            "RUNFILES_DIR",
            "JAVA_RUNFILES",
        ]),
        env_add={"TEST_SRCDIR": "__ignore_me__"})
    self.AssertExitCode(exit_code, 0, stderr)
    if len(stdout) < 2:
      self.fail("stdout: %s" % stdout)
    self.assertEqual(stdout[0], "Hello Python Bar!")
    six.assertRegex(self, stdout[1], "^rloc=.*/bar/bar-py-data.txt")
    self.assertNotIn("__ignore_me__", stdout[1])

    with open(stdout[1].split("=", 1)[1], "r") as f:
      lines = [l.strip() for l in f.readlines()]
    if len(lines) != 1:
      self.fail("lines: %s" % lines)
    self.assertEqual(lines[0], "data for bar.py")


if __name__ == "__main__":
  unittest.main()
