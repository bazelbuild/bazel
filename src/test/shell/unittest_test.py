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
import textwrap
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
    if os.path.isfile(xmlfile):
      with open(xmlfile, "r") as f:
        self._xml = f.read()
    else:
      # Unable to read the file, errors will be reported later.
      self._xml = ""

  # Methods to assert on the state of the results.

  def assertLogMessage(self, message):
    self.assertExactlyOneMatch(self._output, message)

  def assertNotLogMessage(self, message):
    self._asserter.assertNotRegex(self._output, message)

  def assertXmlMessage(self, message):
    self.assertExactlyOneMatch(self._xml, message)

  def assertNotXmlMessage(self, message):
    self._asserter.assertNotRegex(self._xml, message)

  def assertSuccess(self, suite_name):
    self._asserter.assertEqual(0, self._return_code,
                               f"Script failed unexpectedly:\n{self._output}")
    self.assertLogMessage(suite_name)
    self.assertXmlMessage("<testsuites [^/]*failures=\"0\"")
    self.assertXmlMessage("<testsuites [^/]*errors=\"0\"")

  def assertNotSuccess(self, suite_name, failures=0, errors=0):
    self._asserter.assertNotEqual(0, self._return_code)
    self.assertLogMessage(suite_name)
    if failures:
      self.assertXmlMessage(f'<testsuites [^/]*failures="{failures}"')
    if errors:
      self.assertXmlMessage(f'<testsuites [^/]*errors="{errors}"')

  def assertTestPassed(self, test_name):
    self.assertLogMessage(f"PASSED: {test_name}")

  def assertTestFailed(self, test_name, message=""):
    self.assertLogMessage(f"{test_name} FAILED: {message}")

  def assertExactlyOneMatch(self, text, pattern):
    self._asserter.assertRegex(text, pattern)
    self._asserter.assertEqual(
        len(re.findall(pattern, text)),
        1,
        msg=f"Found more than 1 match of '{pattern}' in '{text}'")


class UnittestTest(unittest.TestCase):

  def setUp(self):
    """Create a working directory under our temp dir."""
    super(UnittestTest, self).setUp()
    self.work_dir = tempfile.mkdtemp(dir=os.environ["TEST_TMPDIR"])

  def tearDown(self):
    """Clean up the working directory."""
    super(UnittestTest, self).tearDown()
    shutil.rmtree(self.work_dir)

  def write_file(self, filename, contents=""):
    """Write the contents to a file in the workdir."""

    filepath = os.path.join(self.work_dir, filename)
    with open(filepath, "w") as f:
      f.write(_TEST_PREAMBLE.strip())
      f.write(contents)
    os.chmod(filepath, stat.S_IEXEC | stat.S_IWRITE | stat.S_IREAD)

  def find_runfiles(self):
    if "RUNFILES_DIR" in os.environ:
      return os.environ["RUNFILES_DIR"]

    # Fall back to being based on the srcdir.
    if "TEST_SRCDIR" in os.environ:
      return os.environ["TEST_SRCDIR"]

    # Base on the current dir
    return f"{os.getcwd()}/.."

  def execute_test(self, filename, env=None, args=()):
    """Executes the file and stores the results."""

    filepath = os.path.join(self.work_dir, filename)
    xmlfile = os.path.join(self.work_dir, "dummy-testlog.xml")
    test_env = {
        "TEST_TMPDIR": self.work_dir,
        "RUNFILES_DIR": self.find_runfiles(),
        "TEST_SRCDIR": os.environ["TEST_SRCDIR"],
        "XML_OUTPUT_FILE": xmlfile,
    }
    # Add in env, forcing everything to be a string.
    if env:
      for k, v in env.items():
        test_env[k] = str(v)
    completed = subprocess.run(
        [filepath, *args],
        env=test_env,
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

  def test_set_bash_errexit_prints_stack_trace(self):
    self.write_file(
        "thing.sh", """
set -euo pipefail

function helper() {
  echo before
  false
  echo after
}

function test_failure_in_helper() {
  helper
}

run_suite "bash errexit tests"
""")

    result = self.execute_test("thing.sh")
    result.assertNotSuccess("bash errexit tests")
    result.assertTestFailed("test_failure_in_helper")
    result.assertLogMessage(r"./thing.sh:\d*: in call to helper")
    result.assertLogMessage(
        r"./thing.sh:\d*: in call to test_failure_in_helper")

  def test_set_bash_errexit_runs_tear_down(self):
    self.write_file(
        "thing.sh", """
set -euo pipefail

function tear_down() {
  echo "Running tear_down"
}

function testenv_tear_down() {
  echo "Running testenv_tear_down"
}

function test_failure_in_helper() {
  wrong_command
}

run_suite "bash errexit tests"
""")

    result = self.execute_test("thing.sh")
    result.assertNotSuccess("bash errexit tests")
    result.assertTestFailed("test_failure_in_helper")
    result.assertLogMessage("Running tear_down")
    result.assertLogMessage("Running testenv_tear_down")

  def test_set_bash_errexit_pipefail_propagates_failure_through_pipe(self):
    self.write_file(
        "thing.sh", """
set -euo pipefail

function test_pipefail() {
  wrong_command | cat
  echo after
}

run_suite "bash errexit tests"
""")

    result = self.execute_test("thing.sh")
    result.assertNotSuccess("bash errexit tests")
    result.assertTestFailed("test_pipefail")
    result.assertLogMessage("wrong_command: command not found")
    result.assertNotLogMessage("after")

  def test_set_bash_errexit_no_pipefail_ignores_failure_before_pipe(self):
    self.write_file(
        "thing.sh", """
set -eu
set +o pipefail

function test_nopipefail() {
  wrong_command | cat
  echo after
}

run_suite "bash errexit tests"
""")

    result = self.execute_test("thing.sh")
    result.assertSuccess("bash errexit tests")
    result.assertTestPassed("test_nopipefail")
    result.assertLogMessage("wrong_command: command not found")
    result.assertLogMessage("after")

  def test_set_bash_errexit_pipefail_long_testname_succeeds(self):
    test_name = "x" * 1000
    self.write_file(
        "thing.sh", """
set -euo pipefail

function test_%s() {
  :
}

run_suite "bash errexit tests"
""" % test_name)

    result = self.execute_test("thing.sh")
    result.assertSuccess("bash errexit tests")

  def test_empty_test_fails(self):
    self.write_file("thing.sh", """
# No tests present.

run_suite "empty test suite"
""")

    result = self.execute_test("thing.sh")
    result.assertNotSuccess("empty test suite")
    result.assertLogMessage("No tests found.")

  def test_empty_test_succeeds_sharding(self):
    self.write_file(
        "thing.sh", """
# Only one test.
function test_thing() {
  echo
}

run_suite "empty test suite"
""")

    # First shard.
    result = self.execute_test(
        "thing.sh", env={
            "TEST_TOTAL_SHARDS": 2,
            "TEST_SHARD_INDEX": 0,
        })
    result.assertSuccess("empty test suite")
    result.assertLogMessage("No tests executed due to sharding")

    # Second shard.
    result = self.execute_test(
        "thing.sh", env={
            "TEST_TOTAL_SHARDS": 2,
            "TEST_SHARD_INDEX": 1,
        })
    result.assertSuccess("empty test suite")
    result.assertNotLogMessage("No tests")

  def test_filter_runs_only_matching_test(self):
    self.write_file(
        "thing.sh",
        textwrap.dedent("""
        function test_abc() {
          :
        }

        function test_def() {
          echo "running def"
        }

        run_suite "tests to filter"
        """))

    result = self.execute_test(
        "thing.sh", env={"TESTBRIDGE_TEST_ONLY": "test_a*"})

    result.assertSuccess("tests to filter")
    result.assertTestPassed("test_abc")
    result.assertNotLogMessage("running def")

  def test_filter_prefix_match_only_skips_test(self):
    self.write_file(
        "thing.sh",
        textwrap.dedent("""
        function test_abc() {
          echo "running abc"
        }

        run_suite "tests to filter"
        """))

    result = self.execute_test(
        "thing.sh", env={"TESTBRIDGE_TEST_ONLY": "test_a"})

    result.assertNotSuccess("tests to filter")
    result.assertLogMessage("No tests found.")

  def test_filter_multiple_globs_runs_tests_matching_any(self):
    self.write_file(
        "thing.sh",
        textwrap.dedent("""
        function test_abc() {
          echo "running abc"
        }

        function test_def() {
          echo "running def"
        }

        run_suite "tests to filter"
        """))

    result = self.execute_test(
        "thing.sh", env={"TESTBRIDGE_TEST_ONLY": "donotmatch:*a*"})

    result.assertSuccess("tests to filter")
    result.assertTestPassed("test_abc")
    result.assertNotLogMessage("running def")

  def test_filter_character_group_runs_only_matching_tests(self):
    self.write_file(
        "thing.sh",
        textwrap.dedent("""
        function test_aaa() {
          :
        }

        function test_daa() {
          :
        }

        function test_zaa() {
          echo "running zaa"
        }

        run_suite "tests to filter"
        """))

    result = self.execute_test(
        "thing.sh", env={"TESTBRIDGE_TEST_ONLY": "test_[a-f]aa"})

    result.assertSuccess("tests to filter")
    result.assertTestPassed("test_aaa")
    result.assertTestPassed("test_daa")
    result.assertNotLogMessage("running zaa")

  def test_filter_sharded_runs_subset_of_filtered_tests(self):
    for index in range(2):
      with self.subTest(index=index):
        self.__filter_sharded_runs_subset_of_filtered_tests(index)

  def __filter_sharded_runs_subset_of_filtered_tests(self, index):
    self.write_file(
        "thing.sh",
        textwrap.dedent("""
        function test_a0() {
          echo "running a0"
        }

        function test_a1() {
          echo "running a1"
        }

        function test_bb() {
          echo "running bb"
        }

        run_suite "tests to filter"
        """))

    result = self.execute_test(
        "thing.sh",
        env={
            "TESTBRIDGE_TEST_ONLY": "test_a*",
            "TEST_TOTAL_SHARDS": 2,
            "TEST_SHARD_INDEX": index
        })

    result.assertSuccess("tests to filter")
    # The sharding logic is shifted by 1, starts with 2nd shard.
    result.assertTestPassed("test_a" + str(index ^ 1))
    result.assertLogMessage("running a" + str(index ^ 1))
    result.assertNotLogMessage("running a" + str(index))
    result.assertNotLogMessage("running bb")

  def test_arg_runs_only_matching_test_and_issues_warning(self):
    self.write_file(
        "thing.sh",
        textwrap.dedent("""
        function test_abc() {
          :
        }

        function test_def() {
          echo "running def"
        }

        run_suite "tests to filter"
        """))

    result = self.execute_test("thing.sh", args=["test_abc"])

    result.assertSuccess("tests to filter")
    result.assertTestPassed("test_abc")
    result.assertNotLogMessage("running def")
    result.assertLogMessage(
        r"WARNING: Passing test names in arguments \(--test_arg\) is "
        "deprecated, please use --test_filter='test_abc' instead.")

  def test_arg_multiple_tests_issues_warning_with_test_filter_command(self):
    self.write_file(
        "thing.sh",
        textwrap.dedent("""
        function test_abc() {
          :
        }

        function test_def() {
          :
        }

        run_suite "tests to filter"
        """))

    result = self.execute_test("thing.sh", args=["test_abc", "test_def"])

    result.assertSuccess("tests to filter")
    result.assertTestPassed("test_abc")
    result.assertTestPassed("test_def")
    result.assertLogMessage(
        r"WARNING: Passing test names in arguments \(--test_arg\) is "
        "deprecated, please use --test_filter='test_abc:test_def' instead.")

  def test_arg_and_filter_ignores_arg(self):
    self.write_file(
        "thing.sh",
        textwrap.dedent("""
        function test_abc() {
          :
        }

        function test_def() {
          echo "running def"
        }

        run_suite "tests to filter"
        """))

    result = self.execute_test(
        "thing.sh", args=["test_def"], env={"TESTBRIDGE_TEST_ONLY": "test_a*"})

    result.assertSuccess("tests to filter")
    result.assertTestPassed("test_abc")
    result.assertNotLogMessage("running def")
    result.assertLogMessage(
        "WARNING: Both --test_arg and --test_filter specified, ignoring --test_arg"
    )

  def test_custom_ifs_variable_finds_and_runs_test(self):
    for sharded in (False, True):
      for ifs in (r"\t", "t"):
        with self.subTest(ifs=ifs, sharded=sharded):
          self.__custom_ifs_variable_finds_and_runs_test(ifs, sharded)

  def __custom_ifs_variable_finds_and_runs_test(self, ifs, sharded):
    self.write_file(
        "thing.sh",
        textwrap.dedent(r"""
        set -euo pipefail
        IFS=$'%s'
        function test_foo() {
          :
        }

        run_suite "custom IFS test"
        """ % ifs))

    result = self.execute_test(
        "thing.sh",
        env={} if not sharded else {
            "TEST_TOTAL_SHARDS": 2,
            "TEST_SHARD_INDEX": 1
        })

    result.assertSuccess("custom IFS test")
    result.assertTestPassed("test_foo")

  def test_fail_in_teardown_reports_failure(self):
    self.write_file(
        "thing.sh",
        textwrap.dedent(r"""
        function tear_down() {
          echo "tear_down log" >"${TEST_log}"
          fail "tear_down failure"
        }

        function test_foo() {
          :
        }

        run_suite "Failure in tear_down test"
        """))

    result = self.execute_test("thing.sh")

    result.assertNotSuccess("Failure in tear_down test", errors=1)
    result.assertTestFailed("test_foo", "tear_down failure")
    result.assertXmlMessage('message="tear_down failure"')
    result.assertLogMessage("tear_down log")

  def test_fail_in_teardown_after_test_failure_reports_both_failures(self):
    self.write_file(
        "thing.sh",
        textwrap.dedent(r"""
        function tear_down() {
          echo "tear_down log" >"${TEST_log}"
          fail "tear_down failure"
        }

        function test_foo() {
          echo "test_foo log" >"${TEST_log}"
          fail "Test failure"
        }

        run_suite "Failure in tear_down test"
        """))

    result = self.execute_test("thing.sh")

    result.assertNotSuccess("Failure in tear_down test", errors=1)
    result.assertTestFailed("test_foo", "Test failure")
    result.assertTestFailed("test_foo", "tear_down failure")
    result.assertXmlMessage('message="Test failure"')
    result.assertNotXmlMessage('message="tear_down failure"')
    result.assertXmlMessage("test_foo log")
    result.assertXmlMessage("tear_down log")
    result.assertLogMessage("Test failure")
    result.assertLogMessage("tear_down failure")
    result.assertLogMessage("test_foo log")
    result.assertLogMessage("tear_down log")

  def test_errexit_in_teardown_reports_failure(self):
    self.write_file(
        "thing.sh",
        textwrap.dedent(r"""
        set -euo pipefail

        function tear_down() {
          invalid_command
        }

        function test_foo() {
          :
        }

        run_suite "errexit in tear_down test"
        """))

    result = self.execute_test("thing.sh")

    result.assertNotSuccess("errexit in tear_down test")
    result.assertLogMessage("invalid_command: command not found")
    result.assertXmlMessage('message="No failure message"')
    result.assertXmlMessage("invalid_command: command not found")

  def test_fail_in_tear_down_after_errexit_reports_both_failures(self):
    self.write_file(
        "thing.sh",
        textwrap.dedent(r"""
        set -euo pipefail

        function tear_down() {
          echo "tear_down log" >"${TEST_log}"
          fail "tear_down failure"
        }

        function test_foo() {
          invalid_command
        }

        run_suite "fail after failure"
        """))

    result = self.execute_test("thing.sh")

    result.assertNotSuccess("fail after failure")
    result.assertTestFailed(
        "test_foo",
        "terminated because this command returned a non-zero status")
    result.assertTestFailed("test_foo", "tear_down failure")
    result.assertLogMessage("invalid_command: command not found")
    result.assertLogMessage("tear_down log")
    result.assertXmlMessage('message="No failure message"')
    result.assertXmlMessage("invalid_command: command not found")

  def test_errexit_in_tear_down_after_errexit_reports_both_failures(self):
    self.write_file(
        "thing.sh",
        textwrap.dedent(r"""
        set -euo pipefail

        function tear_down() {
          invalid_command_tear_down
        }

        function test_foo() {
          invalid_command_test
        }

        run_suite "fail after failure"
        """))

    result = self.execute_test("thing.sh")

    result.assertNotSuccess("fail after failure")
    result.assertTestFailed(
        "test_foo",
        "terminated because this command returned a non-zero status")
    result.assertLogMessage("invalid_command_test: command not found")
    result.assertLogMessage("invalid_command_tear_down: command not found")
    result.assertXmlMessage('message="No failure message"')
    result.assertXmlMessage("invalid_command_test: command not found")
    result.assertXmlMessage("invalid_command_tear_down: command not found")


if __name__ == "__main__":
  unittest.main()
