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

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

readonly WRAPPER="${bazel_data}/src/main/tools/process-wrapper"
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
  $WRAPPER -1 0 $OUT $ERR /bin/echo hi there || fail
  assert_output "hi there" ""
}

function test_to_stderr() {
  $WRAPPER -1 0 $OUT $ERR /bin/bash -c "/bin/echo hi there >&2" || fail
  assert_output "" "hi there"
}

function test_exit_code() {
  $WRAPPER -1 0 $OUT $ERR /bin/bash -c "exit 71" || code=$?
  assert_equals 71 "$code"
}

function test_signal_death() {
  $WRAPPER -1 0 $OUT $ERR /bin/bash -c 'kill -ABRT $$' || code=$?
  assert_equals 134 "$code" # SIGNAL_BASE + SIGABRT = 128 + 6
}

function test_signal_catcher() {
  $WRAPPER 2 3 $OUT $ERR /bin/bash -c \
    'trap "echo later; exit 0" SIGINT SIGTERM SIGALRM; sleep 1000' || code=$?
  assert_equals 142 "$code" # SIGNAL_BASE + SIGALRM = 128 + 14
  assert_stdout "later"
}

function test_basic_timeout() {
  $WRAPPER 3 3 $OUT $ERR /bin/bash -c "echo before; sleep 1000; echo after" && fail
  assert_stdout "before"
}

function test_timeout_grace() {
  $WRAPPER 2 3 $OUT $ERR /bin/bash -c \
    'trap "echo -n before; sleep 1; echo -n after; exit 0" SIGINT SIGTERM SIGALRM; sleep 1000' || code=$?
  assert_equals 142 "$code" # SIGNAL_BASE + SIGALRM = 128 + 14
  assert_stdout "beforeafter"
}

function test_timeout_kill() {
  $WRAPPER 2 3 $OUT $ERR /bin/bash -c \
    'trap "echo before; sleep 1000; echo after; exit 0" SIGINT SIGTERM SIGALRM; sleep 1000' || code=$?
  assert_equals 142 "$code" # SIGNAL_BASE + SIGALRM = 128 + 14
  assert_stdout "before"
}

run_suite "process-wrapper"
