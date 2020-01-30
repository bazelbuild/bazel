# Copyright 2020 The Bazel Authors. All rights reserved.
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

# Lint as: python3
"""Tests for unittest.bash."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest

# The test setup for this external test is forwarded to the internal bash test.
# This allows the internal test to use the same runfiles to load unittest.bash.
_TEST_PREAMBLE = """
#!/bin/bash
# --- begin runfiles.bash initialization ---
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

echo "Writing XML to ${XML_OUTPUT_FILE}"

source "$(rlocation "io_bazel/src/test/shell/unittest.bash")" \
  || { echo "Could not source unittest.bash" >&2; exit 1; }
"""

ANSI_ESCAPE = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")


def remove_ansi(line):
  """Remove ANSI-style escape sequences from the input."""
  return ANSI_ESCAPE.sub("", line)


class TestResult(object):
  """Save test results for easy checking."""

  def __init__(self, asserter, return_code, output, xmlfile):
    self._asserter = asserter
    self._return_code = return_code
    self._output = remove_ansi(output)

    # Read in the XML result file.
    with open(xmlfile, "r") as f:
      self._xml = f.read()

  # Methods to assert on the state of the results.

  def assertLogMessage(self, message):
    self._asserter.assertRegex(self._output, message)

  def assertXmlMessage(self, message):
    self._asserter.assertRegex(self._xml, message)

  def assertSuccess(self, suite_name):
    self._asserter.assertEqual(0, self._return_code)
    self.assertLogMessage(suite_name)
    self.assertXmlMessage("failures=\"0\"")
    self.assertXmlMessage("errors=\"0\"")

  def assertNotSuccess(self, suite_name, failures=0, errors=0):
    self._asserter.assertNotEqual(0, self._return_code)
    self.assertLogMessage(suite_name)
    if failures:
      self.assertXmlMessage("failures=\"%d\"" % failures)
    if errors:
      self.assertXmlMessage("errors=\"%d\"" % errors)

  def assertTestPassed(self, test_name):
    self.assertLogMessage("PASSED: %s" % test_name)

  def assertTestFailed(self, test_name, message=""):
    self.assertLogMessage("%s FAILED" % test_name)
    if message:
      self.assertLogMessage("FAILED: %s" % message)


class UnittestTest(unittest.TestCase):

  def setUp(self):
    """Create a working directory under our temp dir."""
    self.work_dir = tempfile.mkdtemp(dir=os.environ['TEST_TMPDIR'])

  def tearDown(self):
    """Clean up the working directory."""
    shutil.rmtree(self.work_dir)

  def write_file(self, filename, contents=""):
    """Write the contents to a file in the workdir."""

    filepath = os.path.join(self.work_dir, filename)
    with open(filepath, "w") as f:
      f.write(_TEST_PREAMBLE.strip())
      f.write(contents)
    os.chmod(filepath, stat.S_IEXEC | stat.S_IWRITE | stat.S_IREAD)

  def execute_test(self, filename):
    """Executes the file and stores the results."""

    filepath = os.path.join(self.work_dir, filename)
    xmlfile = os.path.join(self.work_dir, "dummy-testlog.xml")
    completed = subprocess.run(
        [filepath],
        env={
            "TEST_TMPDIR": self.work_dir,
            "RUNFILES_DIR": os.environ["RUNFILES_DIR"],
            "XML_OUTPUT_FILE": xmlfile,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return TestResult(self, completed.returncode,
                      completed.stdout.decode("utf-8"), xmlfile)

  # Actual test cases.

  def test_success(self):
    self.write_file(
        "thing.sh", """
function test_success() {
  echo foo >&${TEST_log} || fail "expected echo to succeed"
  expect_log "foo"
}

run_suite "success tests"
""")

    result = self.execute_test("thing.sh")
    result.assertSuccess("success tests")
    result.assertTestPassed("test_success")

  def test_timestamp(self):
    self.write_file(
        "thing.sh", """
function test_timestamp() {
  local ts=$(timestamp)
  [[ $ts =~ ^[0-9]{13}$ ]] || fail "timestamp wan't valid: $ts"

  local time_diff=$(get_run_time 100000 223456)
  assert_equals $time_diff 123.456
}

run_suite "timestamp tests"
""")

    result = self.execute_test("thing.sh")
    result.assertSuccess("timestamp tests")
    result.assertTestPassed("test_timestamp")

  def test_failure(self):
    self.write_file(
        "thing.sh", """
function test_failure() {
  fail "I'm a failure with <>&\\" escaped symbols"
}

run_suite "failure tests"
""")

    result = self.execute_test("thing.sh")
    result.assertNotSuccess("failure tests", failures=0, errors=1)
    result.assertTestFailed("test_failure")
    result.assertXmlMessage(
        "message=\"I'm a failure with &lt;&gt;&amp;&quot; escaped symbols\"")
    result.assertXmlMessage("I'm a failure with <>&\" escaped symbols")

  def test_errexit_prints_stack_trace(self):
    self.write_file(
        "thing.sh", """
enable_errexit

function helper() {
  echo before
  false
  echo after
}

function test_errexit() {
  helper
}

run_suite "errexit tests"
""")

    result = self.execute_test("thing.sh")
    result.assertNotSuccess("errexit tests")
    result.assertTestFailed("test_errexit")
    result.assertLogMessage(r"./thing.sh:[0-9]*: in call to helper")
    result.assertLogMessage(r"./thing.sh:[0-9]*: in call to test_errexit")


if __name__ == "__main__":
  unittest.main()
