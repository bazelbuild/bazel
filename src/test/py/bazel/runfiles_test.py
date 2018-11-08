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

  def _AssertRunfilesLibraryInBazelToolsRepo(self, family, lang_name):
    for s, t, exe in [("WORKSPACE.mock", "WORKSPACE",
                       False), ("foo/BUILD.mock", "foo/BUILD",
                                False), ("foo/foo.py", "foo/foo.py", True),
                      ("foo/Foo.java", "foo/Foo.java",
                       False), ("foo/foo.sh", "foo/foo.sh",
                                True), ("foo/foo.cc", "foo/foo.cc", False),
                      ("foo/datadep/hello.txt", "foo/datadep/hello.txt",
                       False), ("bar/BUILD.mock", "bar/BUILD",
                                False), ("bar/bar.py", "bar/bar.py", True),
                      ("bar/bar-py-data.txt", "bar/bar-py-data.txt",
                       False), ("bar/Bar.java", "bar/Bar.java",
                                False), ("bar/bar-java-data.txt",
                                         "bar/bar-java-data.txt", False),
                      ("bar/bar.sh", "bar/bar.sh",
                       True), ("bar/bar-sh-data.txt", "bar/bar-sh-data.txt",
                               False), ("bar/bar.cc", "bar/bar.cc",
                                        False), ("bar/bar-cc-data.txt",
                                                 "bar/bar-cc-data.txt", False)]:
      self.CopyFile(
          self.Rlocation("io_bazel/src/test/py/bazel/testdata/runfiles_test/" +
                         s), t, exe)

    exit_code, stdout, stderr = self.RunBazel(["info", "bazel-bin"])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    exit_code, _, stderr = self.RunBazel([
        "build", "--verbose_failures", "//foo:runfiles-" + family
    ])
    self.AssertExitCode(exit_code, 0, stderr)

    if test_base.TestBase.IsWindows():
      bin_path = os.path.join(bazel_bin, "foo/runfiles-%s.exe" % family)
    else:
      bin_path = os.path.join(bazel_bin, "foo/runfiles-" + family)

    self.assertTrue(os.path.exists(bin_path))

    exit_code, stdout, stderr = self.RunProgram(
        [bin_path], env_add={"TEST_SRCDIR": "__ignore_me__"})
    self.AssertExitCode(exit_code, 0, stderr)
    # 10 output lines: 2 from foo-<family>, and 2 from each of bar-<lang>.
    if len(stdout) != 10:
      self.fail("stdout: %s" % stdout)

    self.assertEqual(stdout[0], "Hello %s Foo!" % lang_name)
    six.assertRegex(self, stdout[1], "^rloc=.*/foo/datadep/hello.txt")
    self.assertNotIn("__ignore_me__", stdout[1])

    with open(stdout[1].split("=", 1)[1], "r") as f:
      lines = [l.strip() for l in f.readlines()]
    if len(lines) != 1:
      self.fail("lines: %s" % lines)
    self.assertEqual(lines[0], "world")

    i = 2
    for lang in [("py", "Python", "bar.py"), ("java", "Java", "Bar.java"),
                 ("sh", "Bash", "bar.sh"), ("cc", "C++", "bar.cc")]:
      self.assertEqual(stdout[i], "Hello %s Bar!" % lang[1])
      six.assertRegex(self, stdout[i + 1],
                      "^rloc=.*/bar/bar-%s-data.txt" % lang[0])
      self.assertNotIn("__ignore_me__", stdout[i + 1])

      with open(stdout[i + 1].split("=", 1)[1], "r") as f:
        lines = [l.strip() for l in f.readlines()]
      if len(lines) != 1:
        self.fail("lines(%s): %s" % (lang[0], lines))
      self.assertEqual(lines[0], "data for " + lang[2])

      i += 2

  def testPythonRunfilesLibraryInBazelToolsRepo(self):
    self._AssertRunfilesLibraryInBazelToolsRepo("py", "Python")

  def testJavaRunfilesLibraryInBazelToolsRepo(self):
    self._AssertRunfilesLibraryInBazelToolsRepo("java", "Java")

  def testBashRunfilesLibraryInBazelToolsRepo(self):
    self._AssertRunfilesLibraryInBazelToolsRepo("sh", "Bash")

  def testCppRunfilesLibraryInBazelToolsRepo(self):
    self._AssertRunfilesLibraryInBazelToolsRepo("cc", "C++")

  def testRunfilesLibrariesFindRunfilesWithoutEnvvars(self):
    for s, t, exe in [
        ("WORKSPACE.mock", "WORKSPACE", False),
        ("bar/BUILD.mock", "bar/BUILD", False),
        ("bar/bar.py", "bar/bar.py", True),
        ("bar/bar-py-data.txt", "bar/bar-py-data.txt", False),
        ("bar/Bar.java", "bar/Bar.java", False),
        ("bar/bar-java-data.txt", "bar/bar-java-data.txt", False),
        ("bar/bar.sh", "bar/bar.sh", True),
        ("bar/bar-sh-data.txt", "bar/bar-sh-data.txt", False),
        ("bar/bar.cc", "bar/bar.cc", False),
        ("bar/bar-cc-data.txt", "bar/bar-cc-data.txt", False),
    ]:
      self.CopyFile(
          self.Rlocation("io_bazel/src/test/py/bazel/testdata/runfiles_test/" +
                         s), t, exe)

    exit_code, stdout, stderr = self.RunBazel(["info", "bazel-bin"])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    exit_code, _, stderr = self.RunBazel([
        "build", "--verbose_failures",
        "//bar:bar-py", "//bar:bar-java", "//bar:bar-sh", "//bar:bar-cc"
    ])
    self.AssertExitCode(exit_code, 0, stderr)

    for lang in [("py", "Python", "bar.py"), ("java", "Java", "Bar.java"),
                 ("sh", "Bash", "bar.sh"), ("cc", "C++", "bar.cc")]:
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
    for s, t, exe in [
        ("WORKSPACE.mock", "WORKSPACE", False),
        ("bar/BUILD.mock", "bar/BUILD", False),
        # Note: do not test Python here, because py_binary always needs a
        # runfiles tree, even on Windows, because it needs __init__.py files in
        # every directory where there may be importable modules, so Bazel always
        # needs to create a runfiles tree for py_binary.
        ("bar/Bar.java", "bar/Bar.java", False),
        ("bar/bar-java-data.txt", "bar/bar-java-data.txt", False),
        ("bar/bar.sh", "bar/bar.sh", True),
        ("bar/bar-sh-data.txt", "bar/bar-sh-data.txt", False),
        ("bar/bar.cc", "bar/bar.cc", False),
        ("bar/bar-cc-data.txt", "bar/bar-cc-data.txt", False),
    ]:
      self.CopyFile(
          self.Rlocation("io_bazel/src/test/py/bazel/testdata/runfiles_test/" +
                         s), t, exe)

    exit_code, stdout, stderr = self.RunBazel(["info", "bazel-bin"])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = stdout[0]

    for lang in [("java", "Java"), ("sh", "Bash"), ("cc", "C++")]:
      exit_code, _, stderr = self.RunBazel([
          "build", "--verbose_failures",
          "--enable_runfiles=no", "//bar:bar-" + lang[0]
      ])
      self.AssertExitCode(exit_code, 0, stderr)

      if test_base.TestBase.IsWindows():
        bin_path = os.path.join(bazel_bin, "bar/bar-%s.exe" % lang[0])
      else:
        bin_path = os.path.join(bazel_bin, "bar/bar-" + lang[0])

      manifest_path = bin_path + ".runfiles_manifest"
      self.assertTrue(os.path.exists(bin_path))
      self.assertTrue(os.path.exists(manifest_path))

      # Create a copy of the runfiles manifest, replacing
      # "bar/bar-<lang>-data.txt" with a custom file.
      mock_bar_dep = self.ScratchFile("bar-%s-mockdata.txt" % lang[0],
                                      ["mock %s data" % lang[0]])
      if test_base.TestBase.IsWindows():
        # Runfiles manifests use forward slashes as path separators, even on
        # Windows.
        mock_bar_dep = mock_bar_dep.replace("\\", "/")
      manifest_key = "foo_ws/bar/bar-%s-data.txt" % lang[0]
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

      substitute_manifest = self.ScratchFile(
          "mock-%s.runfiles/MANIFEST" % lang[0], mock_manifest_data)

      exit_code, stdout, stderr = self.RunProgram(
          [bin_path],
          env_remove=set(["RUNFILES_DIR"]),
          env_add={
              # On Linux/macOS, the Java launcher picks up JAVA_RUNFILES and
              # ignores RUNFILES_MANIFEST_FILE.
              "JAVA_RUNFILES": substitute_manifest[:-len("/MANIFEST")],
              # On Windows, the Java launcher picks up RUNFILES_MANIFEST_FILE.
              # The C++ runfiles library picks up RUNFILES_MANIFEST_FILE on all
              # platforms.
              "RUNFILES_MANIFEST_FILE": substitute_manifest,
              "RUNFILES_MANIFEST_ONLY": "1",
              "TEST_SRCDIR": "__ignore_me__",
          })

      self.AssertExitCode(exit_code, 0, stderr)
      if len(stdout) < 2:
        self.fail("stdout: %s" % stdout)
      self.assertEqual(stdout[0], "Hello %s Bar!" % lang[1])
      six.assertRegex(self, stdout[1], "^rloc=" + mock_bar_dep)
      self.assertNotIn("__ignore_me__", stdout[1])

      with open(stdout[1].split("=", 1)[1], "r") as f:
        lines = [l.strip() for l in f.readlines()]
      if len(lines) != 1:
        self.fail("lines: %s" % lines)
      self.assertEqual(lines[0], "mock %s data" % lang[0])


if __name__ == "__main__":
  unittest.main()
