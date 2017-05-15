#!/bin/bash
#
# Copyright 2015 The Bazel Authors. All rights reserved.
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
# Test sandboxing spawn strategy
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

readonly OUT_DIR="${TEST_TMPDIR}/out"
readonly OUT="${OUT_DIR}/outfile"
readonly ERR="${OUT_DIR}/errfile"

function set_up() {
  rm -rf $OUT_DIR
  mkdir -p $OUT_DIR
}

function assert_stdout() {
  assert_equals "$1" "$(cat $OUT)"
}

function assert_output() {
  assert_equals "$1" "$(cat $OUT)"
  assert_equals "$2" "$(cat $ERR)"
}

function test_basic_functionality() {
  $process_wrapper -1 0 $OUT $ERR /bin/echo hi there &> $TEST_log || fail
  assert_output "hi there" ""
}

function test_to_stderr() {
  $process_wrapper -1 0 $OUT $ERR /bin/bash -c "/bin/echo hi there >&2" &> $TEST_log || fail
  assert_output "" "hi there"
}

function test_exit_code() {
  local code=0
  $process_wrapper -1 0 $OUT $ERR /bin/bash -c "exit 71" &> $TEST_log || code=$?
  assert_equals 71 "$code"
}

function test_signal_death() {
  local code=0
  $process_wrapper -1 0 $OUT $ERR /bin/bash -c 'kill -ABRT $$' &> $TEST_log || code=$?
  assert_equals 134 "$code" # SIGNAL_BASE + SIGABRT = 128 + 6
}

function test_signal_catcher() {
  local code=0
  $process_wrapper 1 2 $OUT $ERR /bin/bash -c \
    'trap "echo later; exit 0" SIGINT SIGTERM SIGALRM; sleep 10' &> $TEST_log || code=$?
  assert_equals 142 "$code" # SIGNAL_BASE + SIGALRM = 128 + 14
  assert_stdout "later"
}

function test_basic_timeout() {
  $process_wrapper 1 2 $OUT $ERR /bin/bash -c "echo before; sleep 10; echo after" &> $TEST_log && fail
  assert_stdout "before"
}

# Tests that process_wrapper sends a SIGTERM to a process on timeout, but gives
# it a grace period of 10 seconds before killing it with SIGKILL.
# In this variant we expect the trap (that runs on SIGTERM) to exit within the
# grace period, thus printing "beforeafter".
function test_timeout_grace() {
  local code=0
  $process_wrapper 1 10 $OUT $ERR /bin/bash -c \
    'trap "echo -n "before"; sleep 1; echo "after"; exit 0" SIGINT SIGTERM SIGALRM; sleep 10' \
    &> $TEST_log || code=$?
  assert_equals 142 "$code" # SIGNAL_BASE + SIGALRM = 128 + 14
  assert_stdout "beforeafter"
}

# Tests that process_wrapper sends a SIGTERM to a process on timeout, but gives
# it a grace period of 2 seconds before killing it with SIGKILL.
# In this variant, we expect the process to be killed with SIGKILL, because the
# trap takes longer than the grace period, thus only printing "before".
function test_timeout_kill() {
  local code=0
  $process_wrapper 1 2 $OUT $ERR /bin/bash -c \
    'trap "echo before; sleep 10; echo after; exit 0" SIGINT SIGTERM SIGALRM; sleep 10' \
    &> $TEST_log || code=$?
  assert_equals 142 "$code" # SIGNAL_BASE + SIGALRM = 128 + 14
  assert_stdout "before"
}

function test_execvp_error_message() {
  local code=0
  $process_wrapper -1 0 $OUT $ERR /bin/notexisting &> $TEST_log || code=$?
  assert_equals 1 "$code"
  assert_contains "execvp(\"/bin/notexisting\", ...): No such file or directory" "$ERR"
}

run_suite "process-wrapper"
