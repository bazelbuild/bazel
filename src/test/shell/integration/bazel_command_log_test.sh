#!/bin/bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

log="$(bazel --batch info command_log)"

function tear_down() {
  # Clean up after ourselves.
  bazel --nobatch shutdown
}

function strip_lines_from_bazel_cc() {
  # sed can't redirect back to its input file (it'll only generate an empty
  # file). In newer versions of gnu sed there is a -i option to edit in place.

  # different sandbox_root result in different startup options
  clean_log=$(\
    sed\
    -e '/^Sending SIGTERM to previous B(l)?aze(l)? server/d'\
    -e "/^INFO: Reading 'startup' options from /d"\
    -e '/^INFO: $TEST_TMPDIR defined: output root default is/d'\
    -e '/^OpenJDK 64-Bit Server VM warning: ignoring option UseSeparateVSpacesInYoungGen; support was removed in 8.0/d'\
    -e '/^Extracting B(l)?aze(l)? installation\.\.\.$/d'\
    -e '/Waiting for response from B(l)?aze(l)? server/d'\
    -e '/^\.*$/d'\
    -e '/^Killed non-responsive server process/d'\
    -e '/server needs to be killed, because the startup options are different/d'\
    $TEST_log)

  echo "$clean_log" > $TEST_log
}

function test_batch_mode() {
  # capture stdout/stderr in $TEST_log
  bazel --batch info >&$TEST_log || fail "Expected success"

  # strip extra lines printed by bazel.cc
  strip_lines_from_bazel_cc

  # compare $TEST_log with command.log
  assert_equals "" "$(diff $TEST_log $log 2>&1)"
}

function test_batch_mode_with_logging_flag() {
  LOG_FILE="$(bazel info output_base)/java.log"
  if [ ! -f $LOG_FILE ]; then
    mkdir -p log_out || fail "Could not create log_out"
    GOOGLE_LOG_DIR=$(pwd)/log_out
    LOG_FILE="log_out/blaze.INFO"
  fi
  bazel --batch info --logging 6 >&$TEST_log || fail "Expected success"

  # strip extra lines printed by bazel.cc
  strip_lines_from_bazel_cc

  # compare $TEST_log with command.log
  assert_equals "" "$(diff $TEST_log $log 2>&1)"

  # verify logging output is captured
  assert_equals "1" $(grep -c "Log level: FINEST$" $LOG_FILE)
}

function test_client_server_mode() {
  # capture stdout/stderr in $TEST_log
  bazel info >&$TEST_log || fail "Expected success"

  # strip extra lines printed by bazel.cc
  strip_lines_from_bazel_cc

  # compare $TEST_log with command.log
  assert_equals "" "$(diff $TEST_log $log 2>&1)"
}

function test_client_server_mode_with_logging_flag() {
  # capture stdout/stderr in $TEST_log
  bazel info --logging 6 >&$TEST_log || fail "Expected success"

  # strip extra lines printed by bazel.cc
  strip_lines_from_bazel_cc

  # compare $TEST_log with command.log
  assert_equals "" "$(diff $TEST_log $log 2>&1)"
}

run_suite "Integration tests of ${PRODUCT_NAME} command log."
