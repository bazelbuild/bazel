#!/bin/bash
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
#
# Tests of the bazel client.

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }


#### TESTS #############################################################

function test_client_debug() {
  # Test that --client_debug sends log statements to stderr.
  bazel --client_debug version >&$TEST_log || fail "'bazel version' failed"
  expect_log "Debug logging requested"
  bazel --client_debug --batch version >&$TEST_log || fail "'bazel version' failed"
  expect_log "Debug logging requested"

  # Test that --client_debug can be disabled
  bazel --noclient_debug version >&$TEST_log || fail "'bazel version' failed"
  expect_not_log "Debug logging requested"
  bazel --noclient_debug --batch version >&$TEST_log || fail "'bazel version' failed"
  expect_not_log "Debug logging requested"

  # Test that --client_debug is off by default.
  bazel --ignore_all_rc_files version >&$TEST_log || fail "'bazel version' failed"
  expect_not_log "Debug logging requested"
  bazel  --ignore_all_rc_files --batch version >&$TEST_log || fail "'bazel version' failed"
  expect_not_log "Debug logging requested"
}

function test_client_debug_change_does_not_restart_server() {
  local server_pid1=$(bazel --client_debug info server_pid 2>$TEST_log)
  local server_pid2=$(bazel info server_pid 2>$TEST_log)
  assert_equals "$server_pid1" "$server_pid2"
  expect_not_log "WARNING.* Running B\\(azel\\|laze\\) server needs to be killed"
}

function test_server_restart_due_to_startup_options() {
  local server_pid1=$(bazel --write_command_log info server_pid 2>$TEST_log)
  local server_pid2=$(bazel --nowrite_command_log info server_pid 2>$TEST_log)
  assert_not_equals "$server_pid1" "$server_pid2" # pid changed.
  expect_log "WARNING.* Running B\\(azel\\|laze\\) server needs to be killed"
}

function test_multiple_requests_same_server() {
  local server_pid1=$(bazel info server_pid 2>$TEST_log)
  local server_pid2=$(bazel info server_pid 2>$TEST_log)
  assert_equals "$server_pid1" "$server_pid2"
  expect_not_log "WARNING.* Running B\\(azel\\|laze\\) server needs to be killed"
}

function test_shutdown() {
  local server_pid1=$(bazel info server_pid 2>$TEST_log)
  bazel shutdown >& $TEST_log || fail "Expected success"
  local server_pid2=$(bazel info server_pid 2>$TEST_log)
  assert_not_equals "$server_pid1" "$server_pid2"
  expect_not_log "WARNING.* Running B\\(azel\\|laze\\) server needs to be killed"
}

function test_server_restart_due_to_startup_options_with_client_debug_information() {
  # Using --write_command_log for no particular reason, if that flag is removed, another startup
  # option will do just fine.
  local server_pid1=$(bazel --client_debug --write_command_log info server_pid 2>$TEST_log)
  local server_pid2=$(bazel --client_debug --nowrite_command_log info server_pid 2>$TEST_log)
  assert_not_equals "$server_pid1" "$server_pid2" # pid changed.
  expect_log "\\[bazel WARNING .*\\] Running B\\(azel\\|laze\\) server needs to be killed"
  expect_log "\\[bazel INFO .*\\] Args from the running server that are not included in the current request:"
  expect_log "\\[bazel INFO .*\\]   --write_command_log"
  expect_log "\\[bazel INFO .*\\] Args from the current request that were not included when creating the server:"
  expect_log "\\[bazel INFO .*\\]   --nowrite_command_log"
}

function test_exit_code() {
  bazel query not_a_query >/dev/null &>$TEST_log &&
      fail "bazel query: expected nonzero exit"
  expect_log "'not_a_query'"
}

function test_output_base() {
  out=$(bazel --output_base=$TEST_TMPDIR/output info output_base 2>$TEST_log)
  assert_equals $TEST_TMPDIR/output "$out"
}

function test_output_base_is_file() {
  bazel --output_base=/dev/null &>$TEST_log && fail "Expected non-zero exit"
  expect_log "FATAL.* Output base directory '/dev/null' could not be created.*exists"
}

function test_cannot_create_output_base() {
  bazel --output_base=/foo &>$TEST_log && fail "Expected non-zero exit"
  expect_log "FATAL.* Output base directory '/foo' could not be created"
}

function test_nonwritable_output_base() {
  bazel --output_base=/ &>$TEST_log && fail "Expected non-zero exit"
  expect_log "FATAL.* Output base directory '/' must be readable and writable."
}

function test_no_arguments() {
  bazel >&$TEST_log || fail "Expected zero exit"
  expect_log "Usage: b\\(laze\\|azel\\)"
}


function test_max_idle_secs() {
  # Remove when https://github.com/bazelbuild/bazel/issues/6773 is fixed.
  bazel shutdown

  local server_pid1=$(bazel --max_idle_secs=1 info server_pid 2>$TEST_log)
  sleep 5
  local server_pid2=$(bazel info server_pid 2>$TEST_log)
  assert_not_equals "$server_pid1" "$server_pid2" # pid changed.
  expect_not_log "WARNING: Running B\\(azel\\|laze\\) server needs to be killed"
}

function test_dashdash_before_command() {
  bazel -- info &>$TEST_log && "Expected failure"
  exitcode=$?
  assert_equals 2 $exitcode
  expect_log "\\[bazel FATAL .*\\] Unknown startup option: '--'."
}

function test_dashdash_after_command() {
  bazel info -- &>$TEST_log || fail "info -- failed"
}

function test_nobatch() {
  local pid1=$(bazel --batch --nobatch info server_pid 2> $TEST_log)
  local pid2=$(bazel --batch --nobatch info server_pid 2> $TEST_log)
  assert_equals "$pid1" "$pid2"
  expect_not_log "WARNING.* Running B\\(azel\\|laze\\) server needs to be killed"
  expect_not_log "WARNING.* --batch mode is deprecated."
}

# Regression test for #1875189, "bazel client should pass through '--help' like
# a command".
function test_bazel_dash_dash_help_is_passed_through() {
  bazel --help >&$TEST_log
  expect_log "Usage: b\\(azel\\|laze\\) <command> <options> ..."
  expect_not_log "Unknown startup option: '--help'."
}

function test_bazel_dash_help() {
  bazel -help >&$TEST_log
  expect_log "Usage: b\\(azel\\|laze\\) <command> <options> ..."
}

function test_bazel_dash_h() {
  bazel -h >&$TEST_log
  expect_log "Usage: b\\(azel\\|laze\\) <command> <options> ..."
}

function test_bazel_dash_s_is_not_parsed() {
  bazel -s --help >&$TEST_log && fail "Expected failure"
  expect_log "Unknown startup option: '-s'."
}

function test_batch() {
  local pid1=$(bazel info server_pid 2> $TEST_log)
  local pid2=$(bazel --batch info server_pid 2> $TEST_log)
  assert_not_equals "$pid1" "$pid2"
  expect_log "WARNING.* Running B\\(azel\\|laze\\) server needs to be killed"
  expect_log "WARNING.* --batch mode is deprecated."
}

function test_cmdline_not_written_in_batch_mode() {
  OUTPUT_BASE=$(bazel --batch info output_base 2> $TEST_log)
  rm -f $OUTPUT_BASE/server/cmdline
  OUTPUT_BASE2=$(bazel --batch info output_base 2> $TEST_log)
  assert_equals "$OUTPUT_BASE" "$OUTPUT_BASE2"
  [[ ! -e $OUTPUT_BASE/server/cmdline ]] || fail "Command line file written."
}

function test_bad_command_batch() {
  bazel --batch notacommand &> $TEST_log && "Expected failure"
  exitcode=$?
  assert_equals 2 "$exitcode"
  expect_log "Command 'notacommand' not found."
}

function test_bad_command_nobatch() {
  bazel --nobatch notacommand &> $TEST_log && "Expected failure"
  exitcode=$?
  assert_equals 2 "$exitcode"
  expect_log "Command 'notacommand' not found."
}

run_suite "Tests of the bazel client."
