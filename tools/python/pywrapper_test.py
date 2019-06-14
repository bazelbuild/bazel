# pylint: disable=g-bad-file-header
# Copyright 2019 The Bazel Authors. All rights reserved.
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

from __future__ import print_function

import os
import subprocess
import textwrap
import unittest

from src.test.py.bazel import test_base


class MockPythonLines(object):

  NORMAL = textwrap.dedent(r"""\
      if [ "$1" = "-V" ]; then
          echo "Mock Python 2.xyz!"
      else
          echo "I am mock Python!"
      fi
      """).split("\n")

  FAIL = textwrap.dedent(r"""\
      echo "Mock failure!"
      exit 1
      """).split("\n")

  WRONG_VERSION = textwrap.dedent(r"""\
      if [ "$1" = "-V" ]; then
          echo "Mock Python 3.xyz!"
      else
          echo "I am mock Python!"
      fi
      """).split("\n")

  VERSION_ERROR = textwrap.dedent(r"""\
      if [ "$1" = "-V" ]; then
          echo "Error!"
          exit 1
      else
          echo "I am mock Python!"
      fi
      """).split("\n")


# TODO(brandjon): Switch to shutil.which when the test is moved to PY3.
def which(cmd):
  """A poor man's approximation of `shutil.which()` or the `which` command.

  Args:
      cmd: The command (executable) name to lookup; should not contain path
        separators

  Returns:
      The absolute path to the first match in PATH, or None if not found.
  """
  for p in os.environ["PATH"].split(os.pathsep):
    fullpath = os.path.abspath(os.path.join(p, cmd))
    if os.path.exists(fullpath):
      return fullpath
  return None


# TODO(brandjon): Move this test to PY3. Blocked (ironically!) on the fix for
# #4815 being available in the host version of Bazel used to run this test.
class PywrapperTest(test_base.TestBase):
  """Unit tests for pywrapper_template.txt.

  These tests are based on the instantiation of the template for Python 2. They
  ensure that the wrapper can locate, validate, and launch a Python 2 executable
  on PATH. To ensure hermeticity, the tests launch the wrapper with PATH
  restricted to the scratch directory.

  Unix only.
  """

  def setup_tool(self, cmd):
    """Copies a command from its system location to the test directory."""
    path = which(cmd)
    self.assertIsNotNone(
        path, msg="Could not locate '%s' command on PATH" % cmd)
    self.CopyFile(path, os.path.join("dir", cmd), executable=True)

  def locate_runfile(self, runfile_path):
    resolved_path = self.Rlocation(runfile_path)
    self.assertIsNotNone(
        resolved_path, msg="Could not locate %s in runfiles" % runfile_path)
    return resolved_path

  def setUp(self):
    super(PywrapperTest, self).setUp()

    # Locate scripts under test.
    self.wrapper_path = \
        self.locate_runfile("io_bazel/tools/python/py2wrapper.sh")
    self.nonstrict_wrapper_path = \
        self.locate_runfile("io_bazel/tools/python/py2wrapper_nonstrict.sh")

    # Setup scratch directory with all executables the script depends on.
    #
    # This is brittle, but we need to make sure we can run the script when only
    # the scratch directory is on PATH, so that we can control whether or not
    # the python executables exist on PATH.
    self.setup_tool("which")
    self.setup_tool("echo")
    self.setup_tool("grep")

  def run_with_restricted_path(self, program, title_for_logging=None):
    new_env = dict(os.environ)
    new_env["PATH"] = self.Path("dir")
    proc = subprocess.Popen([program],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True,
                            cwd=self.Path("dir"),
                            env=new_env)
    # TODO(brandjon): Add a timeout arg here when upgraded to PY3.
    out, err = proc.communicate()
    if title_for_logging is not None:
      print(textwrap.dedent("""\
          ----------------
          %s
          Exit code: %d
          stdout:
          %s
          stderr:
          %s
          ----------------
          """) % (title_for_logging, proc.returncode, out, err))
    return proc.returncode, out, err

  def run_wrapper(self, title_for_logging):
    return self.run_with_restricted_path(self.wrapper_path, title_for_logging)

  def run_nonstrict_wrapper(self, title_for_logging):
    return self.run_with_restricted_path(self.nonstrict_wrapper_path,
                                         title_for_logging)

  def assert_wrapper_success(self, returncode, out, err):
    self.assertEqual(returncode, 0, msg="Expected to exit without error")
    self.assertEqual(
        out, "I am mock Python!\n", msg="stdout was not as expected")
    self.assertEqual(err, "", msg="Expected to produce no stderr output")

  def assert_wrapper_failure(self, returncode, out, err, message):
    self.assertEqual(returncode, 1, msg="Expected to exit with error code 1")
    self.assertRegexpMatches(
        err, message, msg="stderr did not contain expected string")

  def test_finds_python2(self):
    self.ScratchFile("dir/python2", MockPythonLines.NORMAL, executable=True)
    returncode, out, err = self.run_wrapper("test_finds_python2")
    self.assert_wrapper_success(returncode, out, err)

  def test_finds_python(self):
    self.ScratchFile("dir/python", MockPythonLines.NORMAL, executable=True)
    returncode, out, err = self.run_wrapper("test_finds_python")
    self.assert_wrapper_success(returncode, out, err)

  def test_prefers_python2(self):
    self.ScratchFile("dir/python2", MockPythonLines.NORMAL, executable=True)
    self.ScratchFile("dir/python", MockPythonLines.FAIL, executable=True)
    returncode, out, err = self.run_wrapper("test_prefers_python2")
    self.assert_wrapper_success(returncode, out, err)

  def test_no_interpreter_found(self):
    returncode, out, err = self.run_wrapper("test_no_interpreter_found")
    self.assert_wrapper_failure(returncode, out, err,
                                "Neither 'python2' nor 'python' were found")

  def test_wrong_version(self):
    self.ScratchFile(
        "dir/python2", MockPythonLines.WRONG_VERSION, executable=True)
    returncode, out, err = self.run_wrapper("test_wrong_version")
    self.assert_wrapper_failure(
        returncode, out, err,
        "version is 'Mock Python 3.xyz!', but we need version 2")

  def test_error_getting_version(self):
    self.ScratchFile(
        "dir/python2", MockPythonLines.VERSION_ERROR, executable=True)
    returncode, out, err = self.run_wrapper("test_error_getting_version")
    self.assert_wrapper_failure(returncode, out, err,
                                "Could not get interpreter version")

  def test_interpreter_not_executable(self):
    self.ScratchFile(
        "dir/python2", MockPythonLines.VERSION_ERROR, executable=False)
    returncode, out, err = self.run_wrapper("test_interpreter_not_executable")
    self.assert_wrapper_failure(returncode, out, err,
                                "Neither 'python2' nor 'python' were found")

  def test_wrong_version_ok_for_nonstrict(self):
    self.ScratchFile(
        "dir/python2", MockPythonLines.WRONG_VERSION, executable=True)
    returncode, out, err = \
        self.run_nonstrict_wrapper("test_wrong_version_ok_for_nonstrict")
    self.assert_wrapper_success(returncode, out, err)


if __name__ == "__main__":
  unittest.main()
