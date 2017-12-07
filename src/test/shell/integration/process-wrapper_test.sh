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

set -euo pipefail

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

enable_errexit

readonly STATS_PROTO_PATH="${CURRENT_DIR}/../../../main/protobuf/execution_statistics.proto"
readonly STATS_PROTO_DIR="$(cd "$(dirname "${STATS_PROTO_PATH}")" && pwd)"

readonly CPU_TIME_SPENDER="${CURRENT_DIR}/../../../test/shell/integration/spend_cpu_time"

readonly OUT_DIR="${TEST_TMPDIR}/out"
readonly OUT="${OUT_DIR}/outfile"
readonly ERR="${OUT_DIR}/errfile"

readonly EXIT_STATUS_SIGALRM=$((128 + 14))

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
  $process_wrapper --stdout=$OUT --stderr=$ERR /bin/echo hi there &> $TEST_log || fail
  assert_output "hi there" ""
}

function test_to_stderr() {
  $process_wrapper --stdout=$OUT --stderr=$ERR /bin/sh -c "/bin/echo hi there >&2" &> $TEST_log || fail
  assert_output "" "hi there"
}

function test_exit_code() {
  local code=0
  $process_wrapper --stdout=$OUT --stderr=$ERR /bin/sh -c "exit 71" &> $TEST_log || code=$?
  assert_equals 71 "$code"
}

function test_signal_death() {
  local code=0
  $process_wrapper --stdout=$OUT --stderr=$ERR /bin/sh -c 'kill -ABRT $$' &> $TEST_log || code=$?
  assert_equals 134 "$code" # SIGNAL_BASE + SIGABRT = 128 + 6
}

function test_signal_catcher() {
  local code=0
  $process_wrapper --timeout=1 --kill_delay=10 --stdout=$OUT --stderr=$ERR /bin/sh -c \
    'trap "echo later; exit 0" INT TERM ALRM; sleep 10' &> $TEST_log || code=$?
  assert_equals "${EXIT_STATUS_SIGALRM}" "$code"
  assert_stdout "later"
}

function test_basic_timeout() {
  $process_wrapper --timeout=1 --kill_delay=2 --stdout=$OUT --stderr=$ERR /bin/sh -c \
    "echo before; sleep 10; echo after" &> $TEST_log && fail
  assert_stdout "before"
}

# Tests that process_wrapper sends a SIGTERM to a process on timeout, but gives
# it a grace period of 10 seconds before killing it with SIGKILL.
# In this variant we expect the trap (that runs on SIGTERM) to exit within the
# grace period, thus printing "beforeafter".
function test_timeout_grace() {
  local code=0
  $process_wrapper --timeout=1 --kill_delay=10 --stdout=$OUT --stderr=$ERR /bin/sh -c \
    'trap "echo before; sleep 1; echo after; exit 0" INT TERM ALRM; sleep 10' \
    &> $TEST_log || code=$?
  assert_equals "${EXIT_STATUS_SIGALRM}" "$code"
  assert_stdout 'before
after'
}

# Tests that process_wrapper sends a SIGTERM to a process on timeout, but gives
# it a grace period of 2 seconds before killing it with SIGKILL.
# In this variant, we expect the process to be killed with SIGKILL, because the
# trap takes longer than the grace period, thus only printing "before".
function test_timeout_kill() {
  local code=0
  $process_wrapper --timeout=1 --kill_delay=2 --stdout=$OUT --stderr=$ERR /bin/sh -c \
    'trap "echo before; sleep 10; echo after; exit 0" INT TERM ALRM; sleep 10' \
    &> $TEST_log || code=$?
  assert_equals "${EXIT_STATUS_SIGALRM}" "$code"
  assert_stdout "before"
}

function test_execvp_error_message() {
  local code=0
  $process_wrapper --stdout=$OUT --stderr=$ERR /bin/notexisting &> $TEST_log || code=$?
  assert_equals 1 "$code"
  assert_contains "\"execvp(/bin/notexisting, ...)\": No such file or directory" "$ERR"
}

function check_execution_time_for_command() {
  local user_time_low="$1"; shift 1
  local user_time_high="$1"; shift 1
  local sys_time_low="$1"; shift 1
  local sys_time_high="$1"; shift 1

  local stats_out_path="${OUT_DIR}/statsfile"
  local stats_out_decoded_path="${OUT_DIR}/statsfile.decoded"

  # Wrapped process will be terminated after 100 seconds if not self terminated.
  local code=0
  "${process_wrapper}" --timeout=100 --kill_delay=2 --stdout="${OUT}" \
      --stderr="${ERR}" --stats="${stats_out_path}" \
      "$@" &> "${TEST_log}" || code="$?"
  assert_equals 0 "${code}"

  if [ ! -e "${stats_out_path}" ]; then
    fail "Stats file not found: '${stats_out_path}'"
  fi

  "${protoc_compiler}" --proto_path="${STATS_PROTO_DIR}" \
      --decode tools.protos.ExecutionStatistics execution_statistics.proto \
      < "${stats_out_path}" > "${stats_out_decoded_path}"

  if [ ! -e "${stats_out_decoded_path}" ]; then
    fail "Decoded stats file not found: '${stats_out_decoded_path}'"
  fi

  local utime=0
  if grep -q utime_sec "${stats_out_decoded_path}"; then
    utime="$(grep utime_sec ${stats_out_decoded_path} | cut -f2 -d':' | \
      tr -dc '0-9')"
  fi

  local stime=0
  if grep -q stime_sec "${stats_out_decoded_path}"; then
    stime="$(grep stime_sec ${stats_out_decoded_path} | cut -f2 -d':' | \
      tr -dc '0-9')"
  fi

  if ! [ ${utime} -ge ${user_time_low} -a ${utime} -le ${user_time_high} ]; then
    fail "reported utime of '${utime}' is out of expected range"
  fi
  if ! [ ${stime} -ge ${sys_time_low} -a ${stime} -le ${sys_time_high} ]; then
    fail "reported stime of '${stime}' is out of expected range"
  fi
}

function test_stats_high_user_time() {
  # Tested with blaze test --runs_per_test 1000 on November 28, 2017.
  check_execution_time_for_command 10 11 0 1 \
      "${CPU_TIME_SPENDER}" 10 0
}

function test_stats_high_system_time() {
  # Tested with blaze test --runs_per_test 1000 on November 28, 2017.
  check_execution_time_for_command 0 1 10 11 \
      "${CPU_TIME_SPENDER}" 0 10
}

function test_stats_high_user_time_and_high_system_time() {
  # Tested with blaze test --runs_per_test 1000 on November 28, 2017.
  check_execution_time_for_command 10 11 10 11 \
      "${CPU_TIME_SPENDER}" 10 10
}

run_suite "process-wrapper"
