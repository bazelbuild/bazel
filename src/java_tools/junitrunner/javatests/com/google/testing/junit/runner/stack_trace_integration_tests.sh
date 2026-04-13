#!/usr/bin/env bash
#
# Copyright 2010 The Bazel Authors. All Rights Reserved.
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
# Integration testing of stack traces printed to the console during JUnit
# tests. Test runners send a SIGTERM to the test process upon timeout (and
# typically SIGKILL after a grace period).
#
# These tests operate by executing test methods in a testbed program
# and checking the output.
#

# Load the unit-testing framework
source "$1" || \
  { echo "Failed to load unit-testing framework $1" >&2; exit 1; }

set +o errexit

unset TEST_PREMATURE_EXIT_FILE

TESTBED="${PWD}/$2"
SUITE_PARAMETER="$3"
JUNIT_VERSION="$4"
SUITE="com.google.testing.junit.runner.testbed.StackTraceExercises"
SUITE_FLAG="-D${SUITE_PARAMETER}=${SUITE}"
SLOW_CREATION_SUITE_FLAG="-D${SUITE_PARAMETER}=com.google.testing.junit.runner.testbed.SuiteMethodTakesForever"

if [[ "$SUITE_PARAMETER" == "bazel.test_suite" ]]; then
  is_bazel=true
else
  is_bazel=false
fi

shift 3

# Usage: COUNT=count_in_log <regex>
function count_in_log() {
  echo $(grep -c "$1" $TEST_log)
}

function check_ge() {
  (( $1 >= $2 )) || fail "$3"
}

function check_le() {
  (( $1 <= $2 )) || fail "$3"
}

function check_eq() {
  (( $1 == $2 )) || fail "$3"
}

# Usage: expect_thread_dumps_in_log <max_number_of_expected_thread_dumps>
function expect_thread_dumps_in_log() {
  local thread_dump_starts=$(count_in_log "Starting full thread dump")
  local thread_dump_ends=$(count_in_log "Done full thread dump")
  check_ge "$thread_dump_starts" 1 "Thread dump generated at least once"
  check_le "$thread_dump_starts" "$1" "Thread dump generated at most $1 times"
  check_eq "$thread_dump_starts" "$thread_dump_ends" \
    "Thread dumps ended successfully"
}

# Usage: expect_thread_dumps_in_log <virtual thread dump content>
function expect_json_thread_dumps() {
  local thread_dump_starts=$(count_in_log "Starting full thread dump")
  local json_thread_dump=$(count_in_log "Writing JSON thread dump to ")
  check_eq "$thread_dump_starts" "$json_thread_dump" \
    "JSON thread dumps generated as expected"
  # Extract the path of the JSON thread dumps and check that they exist
  local json_paths=$(grep -oE "Writing JSON thread dump to [^ ]+" $TEST_log | \
    sed 's/Writing JSON thread dump to //')
  for path in $json_paths; do
    [ -f "$path" ] || fail "Expected JSON thread dump file $path to exist"
    grep -q "$1" "$path" || fail "Expected virtual thread dump content not found in $path: $(cat "$path")"
  done
}

#######################

# Test that we see a stack trace even on shutdown hook slowness.
function test_ShutdownHook() {
  cd $TEST_TMPDIR
  local fifo="${PWD}/tmp/fifo_for_shutdown_hook"
  mkdir -p tmp
  mkfifo $fifo || fail "Couldn't create ${fifo}"

  # Run the test in the background. The test process will report success,
  # but hang in the shutdown hook.
  if [ "$JUNIT_VERSION" = "3" ]; then
    TEST_FILTER_FLAG="${SUITE}#testSneakyShutdownHook"
  else
    TEST_FILTER_FLAG="--test_filter=testSneakyShutdownHook"
  fi

  $TESTBED --jvm_flag=${SUITE_FLAG} --jvm_flag=-Dtest.fifo=${fifo} $TEST_FILTER_FLAG \
    >& $TEST_log & test_pid=$!

  echo "Synchronize to the shutdown hook" > $fifo
  expect_log 'OK.*1 test'
  expect_log "Entered shutdown"

  # Send the SIGTERM and wait 3s (generous) for it to be processed:
  kill -TERM $test_pid
  sleep 3

  if $is_bazel; then
    expect_log 'Received SIGTERM, dumping stack traces for all threads'
    expect_json_thread_dumps '"name": "my-virtual-thread",'
  else
    expect_log 'INTERRUPTED TEST: SIGTERM'
  fi
  expect_log 'Shutdown\.runHooks'
  expect_log 'StackTraceExercises\.handleHook'
  expect_log 'Thread\.sleep'

  # expect threads to be dumped at most 2 times
  expect_thread_dumps_in_log 2

  wait $test_pid || fail "Expected process to finish successfully"
}

# Test that we see a stack trace when the test is interrupted during the test
# suite creation phase.
function test_SlowSuite() {
  cd $TEST_TMPDIR
  local fifo="${PWD}/tmp/fifo_for_slow_suite"
  mkdir -p tmp
  mkfifo $fifo || fail "Couldn't create ${fifo}"

  # Run the test in the background. The test process will hang in the suite
  # creation phase.
  $TESTBED --jvm_flag=${SLOW_CREATION_SUITE_FLAG} --jvm_flag=-Dtest.fifo=${fifo} \
    >& $TEST_log & test_pid=$!
  echo "Synchronize to the suite creation" > $fifo
  expect_log 'Entered suite creation'

  # Send the SIGTERM and wait 3s (generous) for it to be processed:
  kill -TERM $test_pid
  sleep 3

  if $is_bazel; then
    expect_log 'Received SIGTERM, dumping stack traces for all threads'
    expect_json_thread_dumps '"name": "my-virtual-thread",'
  else
    expect_log "Execution interrupted while running 'TestSuite creation'"
  fi

  # expect threads to be dumped exactly once
  expect_thread_dumps_in_log 1
}

# If a test calls System.exit(), make sure it leaves the
# TEST_PREMATURE_EXIT_FILE around.
function test_PrematureExit() {
  cd $TEST_TMPDIR
  mkdir foo
  local no_exit="${PWD}/foo/premature_exit_file"

  [ ! -a "$no_exit" ] || fail "${no_exit} should not exist yet"

  if [ "$JUNIT_VERSION" = "3" ]; then
    TEST_FILTER_FLAG="${SUITE}#testNotSoFastBuddy"
  else
    TEST_FILTER_FLAG="--test_filter=testNotSoFastBuddy"
  fi

  echo "Redirecting output to $TEST_log"
  TEST_PREMATURE_EXIT_FILE=${no_exit} $TESTBED --jvm_flag=${SUITE_FLAG} $TEST_FILTER_FLAG \
    >& $TEST_log

  expect_log 'Hey, not so fast there'
  [ -r "$no_exit" ] || fail "$no_exit is not readable"
}

run_suite "stacktrace"
