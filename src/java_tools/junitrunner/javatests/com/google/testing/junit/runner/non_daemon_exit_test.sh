#!/usr/bin/env bash
#
# Copyright 2015 The Bazel Authors. All Rights Reserved.
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
#
# Integration testing of the test bridge protocol for JUnit4 tests.
# This is a test filter passed by "bazel test --test_filter".
#
# These tests operate by executing test methods in a testbed program
# and checking the output.
#

[ -z "$TEST_SRCDIR" ] && { echo "TEST_SRCDIR not set!" >&2; exit 1; }

# Load the unit-testing framework
source "$1" || \
  { echo "Failed to load unit-testing framework $1" >&2; exit 1; }

set +o errexit

TESTBED="${PWD}/$2"
SUITE_PARAMETER="$3"
SUITE_FLAG="-D${SUITE_PARAMETER}=com.google.testing.junit.runner.testbed.NonDaemonThreadTest"
XML_OUTPUT_FILE="${TEST_TMPDIR}/test.xml"

shift 3

#######################

function test_JVMDoesNotExitImmediately() {
  cd "${TEST_TMPDIR}" || fail "Unexpected failure"

  # Run the test setting the flag; the JVM should wait for non-daemon to exit
  "${TESTBED}" --jvm_flag=${SUITE_FLAG} --jvm_flag="-Dbazel.test_runner.await_non_daemon_threads=true" >& "${TEST_log}" &
  local testbed_pid=$!

  # sleep something smaller than the 5 second sleep in the test itself
  sleep 1

  if ps -p ${testbed_pid} > /dev/null; then
    echo "TESTBED process is still alive."
  else
    fail "TESTBED process exited too soon, before 5 seconds had elapsed."
  fi

  # sleep something longer than the 5 second sleep in the test itself
  sleep 5
  if ps -p ${testbed_pid} > /dev/null; then
    fail "TESTBED process has not exited yet."
  else
    echo "TESTBED process is no longer alive."
  fi

  # Remove the XML output with failures, so it does not get picked up to
  # indicate a failure.
  rm -rf "${XML_OUTPUT_FILE}" || fail "failed to remove XML output"
}

run_suite "non_daemon_exit_test"
