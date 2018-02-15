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
        ("bar/Bar.java", "bar/Bar.java"),
        ("bar/bar-java-data.txt", "bar/bar-java-data.txt"),
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
    if len(stdout) != 6:
      self.fail("stdout: %s" % stdout)

    self.assertEqual(stdout[0], "Hello Python Foo!")
    six.assertRegex(self, stdout[1], "^rloc=.*/foo/datadep/hello.txt")
    self.assertNotIn("__ignore_me__", stdout[1])

    self.assertEqual(stdout[2], "Hello Python Bar!")
    six.assertRegex(self, stdout[3], "^rloc=.*/bar/bar-py-data.txt")
    self.assertNotIn("__ignore_me__", stdout[3])

    self.assertEqual(stdout[4], "Hello Java Bar!")
    six.assertRegex(self, stdout[5], "^rloc=.*/bar/bar-java-data.txt")
    self.assertNotIn("__ignore_me__", stdout[5])

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

    with open(stdout[5].split("=", 1)[1], "r") as f:
      lines = [l.strip() for l in f.readlines()]
    if len(lines) != 1:
      self.fail("lines: %s" % lines)
    self.assertEqual(lines[0], "data for Bar.java")

  def testRunfilesLibrariesFindRunfilesWithoutEnvvars(self):
    for s, t in [
        ("WORKSPACE.mock", "WORKSPACE"),
        ("bar/BUILD.mock", "bar/BUILD"),
        ("bar/bar.py", "bar/bar.py"),
        ("bar/bar-py-data.txt", "bar/bar-py-data.txt"),
        ("bar/Bar.java", "bar/Bar.java"),
        ("bar/bar-java-data.txt", "bar/bar-java-data.txt"),
    ]:
      self.CopyFile(
          self.Rlocation(
              "io_bazel/src/test/py/bazel/testdata/runfiles_test/" + s), t)

    exit_code, stdout, stderr = self.RunBazel(["info", "bazel-bin"])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    exit_code, _, stderr = self.RunBazel(["build", "//bar:all"])
    self.AssertExitCode(exit_code, 0, stderr)

    for lang in [("py", "Python", "bar.py"), ("java", "Java", "Bar.java")]:
      if test_base.TestBase.IsWindows():
        bin_path = os.path.join(bazel_bin, "bar/bar-%s.exe" % lang[0])
      else:
        bin_path = os.path.join(bazel_bin, "bar/bar-" + lang[0])

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
        self.fail("stdout(%s): %s" % (lang[0], stdout))
      self.assertEqual(stdout[0], "Hello %s Bar!" % lang[1])
      six.assertRegex(self, stdout[1], "^rloc=.*/bar/bar-%s-data.txt" % lang[0])
      self.assertNotIn("__ignore_me__", stdout[1])

      with open(stdout[1].split("=", 1)[1], "r") as f:
        lines = [l.strip() for l in f.readlines()]
      if len(lines) != 1:
        self.fail("lines(%s): %s" % (lang[0], lines))
      self.assertEqual(lines[0], "data for " + lang[2])

  def testRunfilesLibrariesFindRunfilesWithRunfilesManifestEnvvar(self):
    for s, t in [
        ("WORKSPACE.mock", "WORKSPACE"),
        ("bar/BUILD.mock", "bar/BUILD"),
        # Note: do not test Python here, because py_binary always needs a
        # runfiles tree, even on Windows, because it needs __init__.py files in
        # every directory where there may be importable modules, so Bazel always
        # needs to create a runfiles tree for py_binary.
        ("bar/Bar.java", "bar/Bar.java"),
        ("bar/bar-java-data.txt", "bar/bar-java-data.txt"),
    ]:
      self.CopyFile(
          self.Rlocation(
              "io_bazel/src/test/py/bazel/testdata/runfiles_test/" + s), t)

    exit_code, stdout, stderr = self.RunBazel(["info", "bazel-bin"])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    exit_code, _, stderr = self.RunBazel(
        ["build", "--experimental_enable_runfiles=no", "//bar:bar-java"])
    self.AssertExitCode(exit_code, 0, stderr)

    if test_base.TestBase.IsWindows():
      bin_path = os.path.join(bazel_bin, "bar/bar-java.exe")
    else:
      bin_path = os.path.join(bazel_bin, "bar/bar-java")

    manifest_path = bin_path + ".runfiles_manifest"
    self.assertTrue(os.path.exists(bin_path))
    self.assertTrue(os.path.exists(manifest_path))

    # Create a copy of the runfiles manifest, replacing
    # "bar/bar-java-data.txt" with a custom file.
    mock_bar_dep = self.ScratchFile("bar-java-mockdata.txt", ["mock java data"])
    if test_base.TestBase.IsWindows():
      # Runfiles manifests use forward slashes as path separators, even on
      # Windows.
      mock_bar_dep = mock_bar_dep.replace("\\", "/")
    manifest_key = "foo_ws/bar/bar-java-data.txt"
    mock_manifest_line = manifest_key + " " + mock_bar_dep
    with open(manifest_path, "rt") as f:
      # Only rstrip newlines. Do not rstrip() completely, because that would
      # remove spaces too. This is necessary in order to have at least one
      # space in every manifest line.
      # Some manifest entries don't have any path after this space, namely the
      # "__init__.py" entries. (Bazel writes such manifests on every
      # platform). The reason is that these files are never symlinks in the
      # runfiles tree, Bazel actually creates empty __init__.py files (again
      # on every platform). However to keep these manifest entries correct,
      # they need to have a space character.
      # We could probably strip thses lines completely, but this test doesn't
      # aim to exercise what would happen in that case.
      mock_manifest_data = [
          mock_manifest_line
          if line.split(" ", 1)[0] == manifest_key else line.rstrip("\n\r")
          for line in f
      ]

    substitute_manifest = self.ScratchFile("mock-java.runfiles/MANIFEST",
                                           mock_manifest_data)

    exit_code, stdout, stderr = self.RunProgram(
        [bin_path],
        env_remove=set(["RUNFILES_DIR"]),
        env_add={
            # On Linux/macOS, the Java launcher picks up JAVA_RUNFILES and
            # ignores RUNFILES_MANIFEST_FILE.
            "JAVA_RUNFILES": substitute_manifest[:-len("/MANIFEST")],
            # On Windows, the Java launcher picks up RUNFILES_MANIFEST_FILE.
            "RUNFILES_MANIFEST_FILE": substitute_manifest,
            "RUNFILES_MANIFEST_ONLY": "1",
            "TEST_SRCDIR": "__ignore_me__",
        })

    self.AssertExitCode(exit_code, 0, stderr)
    if len(stdout) < 2:
      self.fail("stdout: %s" % stdout)
    self.assertEqual(stdout[0], "Hello Java Bar!")
    six.assertRegex(self, stdout[1], "^rloc=" + mock_bar_dep)
    self.assertNotIn("__ignore_me__", stdout[1])

    with open(stdout[1].split("=", 1)[1], "r") as f:
      lines = [l.strip() for l in f.readlines()]
    if len(lines) != 1:
      self.fail("lines: %s" % lines)
    self.assertEqual(lines[0], "mock java data")


if __name__ == "__main__":
  unittest.main()
