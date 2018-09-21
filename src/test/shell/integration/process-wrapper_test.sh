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

set -euo pipefail

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/execution_statistics_utils.sh" \
  || { echo "execution_statistics_utils.sh not found!" >&2; exit 1; }

enable_errexit

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

function assert_process_wrapper_exec_time() {
  local user_time_low="$1"; shift
  local user_time_high="$1"; shift
  local sys_time_low="$1"; shift
  local sys_time_high="$1"; shift

  local local_tmp="$(mktemp -d "${OUT_DIR}/assert_process_wrapper_timeXXXXXX")"
  local stdout_path="${local_tmp}/stdout"
  local stderr_path="${local_tmp}/stderr"
  local stats_out_path="${local_tmp}/statsfile"
  local stats_out_decoded_path="${local_tmp}/statsfile.decoded"

  # Wrapped process will be terminated after 100 seconds if not self terminated.
  local code=0
  "${process_wrapper}" \
      --timeout=100 \
      --kill_delay=2 \
      --stdout="${stdout_path}" \
      --stderr="${stderr_path}" \
      --stats="${stats_out_path}" \
      "${CPU_TIME_SPENDER}" "${user_time_low}" "${sys_time_low}" \
      &> "${TEST_log}" || code="$?"
  sed -e 's,^subprocess stdout: ,,' "${stdout_path}" >>"${TEST_log}"
  sed -e 's,^subprocess stderr: ,,' "${stderr_path}" >>"${TEST_log}"
  assert_equals 0 "${code}"

  assert_execution_time_in_range \
      "${user_time_low}" \
      "${user_time_high}" \
      "${sys_time_low}" \
      "${sys_time_high}" \
      "${stats_out_path}"
}

function test_stats_high_user_time() {
  assert_process_wrapper_exec_time 10 19 0 9
}

function test_stats_high_system_time() {
  assert_process_wrapper_exec_time 0 9 10 19
}

function test_stats_high_user_time_and_high_system_time() {
  assert_process_wrapper_exec_time 10 25 10 25
}

run_suite "process-wrapper"
