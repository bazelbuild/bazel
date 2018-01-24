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
    self.ScratchFile("WORKSPACE", ["workspace(name = 'foo_ws')"])
    self.ScratchFile("foo/BUILD", [
        "java_binary(",
        "    name = 'Foo',",
        "    main_class = 'Foo',",
        "    srcs = ['Foo.java'],",
        "    deps = ['@bazel_tools//tools/runfiles:java-runfiles'],",
        "    data = ['//foo/bar:hello.txt'],",
        ")"
    ])
    self.ScratchFile("foo/Foo.java", [
        "import com.google.devtools.build.runfiles.Runfiles;",
        ""
        "public class Foo {",
        "  public static void main(String[] args) throws java.io.IOException {",
        "    System.out.println(\"Hello Foo!\");",
        "    Runfiles r = Runfiles.create();",
        "    System.out.println(",
        "        \"rloc=\" + r.rlocation(\"foo_ws/foo/bar/hello.txt\"));",
        "  }",
        "}"
    ])
    self.ScratchFile("foo/bar/BUILD", ["exports_files(['hello.txt'])"])
    self.ScratchFile("foo/bar/hello.txt", ["world"])

    exit_code, stdout, stderr = self.RunBazel(["info", "bazel-bin"])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    exit_code, _, stderr = self.RunBazel(["build", "//foo:Foo"])
    self.AssertExitCode(exit_code, 0, stderr)

    bin_path = os.path.join(bazel_bin, "foo/Foo" +
                            (".exe" if test_base.TestBase.IsWindows() else ""))
    self.assertTrue(os.path.exists(bin_path))

    exit_code, stdout, stderr = self.RunProgram([bin_path])
    self.AssertExitCode(exit_code, 0, stderr)
    if len(stdout) != 2:
      self.fail("stdout: " + stdout)
    self.assertEqual(stdout[0], "Hello Foo!")
    self.assertRegexpMatches(stdout[1], "^rloc=.*/foo/bar/hello.txt")
    with open(stdout[1].split("=", 1)[1], "r") as f:
      lines = [l.strip() for l in f.readlines()]
    if len(lines) != 1:
      self.fail("lines: " + lines)
    self.assertEqual(lines[0], "world")


if __name__ == "__main__":
  unittest.main()
