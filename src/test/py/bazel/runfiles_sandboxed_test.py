# pylint: disable=g-bad-file-header
# Copyright 2018 The Bazel Authors. All rights reserved.
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


class RunfilesSandboxedTest(test_base.TestBase):

  def _FailWithContents(self, msg, contents):
    self.fail("%s\ncontents =\n | %s\n---" % (msg, "\n | ".join(contents)))

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
    self.ScratchFile(
        "foo/BUILD",
        [
            "genrule(",
            "    name = 'gen',",
            "    outs = ['stdout.txt', 'data_files.txt'],",
            "    cmd = 'cat $$(' + ",
            # The genrule runs all bar-<language> tools, saves the complete
            # stdout into stdout.txt, and prints the contents of rlocations
            # reported by the tools (i.e. the contents of the
            # bar-<language>-data.txt files) into data_files.txt.
            "          '  ( $(location //bar:bar-py) && ' +",
            "          '    $(location //bar:bar-java) && ' +",
            "          '    $(location //bar:bar-sh) && ' +",
            "          '    $(location //bar:bar-cc) ; ' +",
            "          '  ) | ' + ",
            "          '    tee $(location stdout.txt) | ' + ",
            "          '    grep \"^rloc=\" | ' + ",
            "          '    sed \"s,^rloc=,,\"' + ",
            "          ') > $(location data_files.txt)',",
            "    tools = [",
            "        '//bar:bar-cc',",
            "        '//bar:bar-java',",
            "        '//bar:bar-py',",
            "        '//bar:bar-sh',",
            "    ],",
            ")"
        ])

    _, stdout, _ = self.RunBazel(["info", "bazel-genfiles"])
    bazel_genfiles = stdout[0]

    self.RunBazel([
        "build",
        "--verbose_failures",
        "//foo:gen",
        "--genrule_strategy=sandboxed",
    ])

    stdout_txt = os.path.join(bazel_genfiles, "foo/stdout.txt")
    self.assertTrue(os.path.isfile(stdout_txt))

    data_files_txt = os.path.join(bazel_genfiles, "foo/data_files.txt")
    self.assertTrue(os.path.isfile(data_files_txt))

    # Output of the bar-<language> binaries that they printed to stdout.
    stdout_lines = []
    with open(stdout_txt, "rt") as f:
      stdout_lines = [line.strip() for line in f.readlines()]

    # Contents of the bar-<language>-data.txt files.
    data_files = []
    with open(data_files_txt, "rt") as f:
      data_files = [line.strip() for line in f.readlines()]

    if len(stdout_lines) != 8:
      self._FailWithContents("wrong number of output lines", stdout_lines)
    i = 0
    for lang in [("py", "Python", "bar.py"), ("java", "Java", "Bar.java"),
                 ("sh", "Bash", "bar.sh"), ("cc", "C++", "bar.cc")]:
      # Check that the bar-<language> binary printed the expected output.
      if stdout_lines[i * 2] != "Hello %s Bar!" % lang[1]:
        self._FailWithContents("wrong line for " + lang[1], stdout_lines)
      if not stdout_lines[i * 2 + 1].startswith("rloc="):
        self._FailWithContents("wrong line for " + lang[1], stdout_lines)
      if not stdout_lines[i * 2 + 1].endswith(
          "foo_ws/bar/bar-%s-data.txt" % lang[0]):
        self._FailWithContents("wrong line for " + lang[1], stdout_lines)

      # Assert the contents of bar-<language>-data.txt. This indicates that
      # the runfiles library in the bar-<language> binary found the correct
      # runfile and returned a valid path.
      if data_files[i] != "data for " + lang[2]:
        self._FailWithContents("runfile does not exist for " + lang[1],
                               stdout_lines)

      i += 1


if __name__ == "__main__":
  unittest.main()
