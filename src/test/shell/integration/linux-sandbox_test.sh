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
source "${CURRENT_DIR}/../sandboxing_test_utils.sh" \
  || { echo "sandboxing_test_utils.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/execution_statistics_utils.sh" \
  || { echo "execution_statistics_utils.sh not found!" >&2; exit 1; }

readonly OUT_DIR="${TEST_TMPDIR}/out"
readonly OUT="${OUT_DIR}/outfile"
readonly ERR="${OUT_DIR}/errfile"
readonly SANDBOX_DIR="${OUT_DIR}/sandbox"
readonly MOUNT_TARGET_ROOT="${TEST_TMPDIR}/targets"

readonly CPU_TIME_SPENDER="${CURRENT_DIR}/../../../test/shell/integration/spend_cpu_time"

SANDBOX_DEFAULT_OPTS="-W $SANDBOX_DIR"

function set_up {
  rm -rf $OUT_DIR
  mkdir -p $SANDBOX_DIR
}

function test_basic_functionality() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -- /bin/echo hi there &> $TEST_log || fail
  expect_log "hi there"
}

function test_execvp_error_message_contains_path() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -- /does/not/exist --hello world &> $TEST_log || code=$?
  expect_log "\"execvp(/does/not/exist, 0x[[:alnum:]]*)\": No such file or directory"
}

function test_default_user_is_current_user() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -- /usr/bin/id &> $TEST_log || fail
  local current_uid_number="$(id -u)"
  # Expecting something like: uid=485038(ruperts) ...
  expect_log "uid=${current_uid_number}("
}

function test_user_switched_to_root() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -R -- /usr/bin/id &> $TEST_log || fail
  expect_log "uid=0(root) gid=0(root) groups=0(root)"
}

function test_user_switched_to_nobody() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -U -- /usr/bin/id &> $TEST_log || fail
  expect_log "uid=[0-9]\+(nobody) gid=[0-9]\+(\(nobody\|nogroup\)) groups=[0-9]\+(\(nobody\|nogroup\))"
}

function test_exit_code() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -- /bin/bash -c "exit 71" &> $TEST_log || code=$?
  assert_equals 71 "$code"
}

function test_signal_death() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -- /bin/bash -c 'kill -ABRT $$' &> $TEST_log || code=$?
  assert_equals 134 "$code" # SIGNAL_BASE + SIGABRT = 128 + 6
}

function test_signal_catcher() {
  # Run the sandbox with a child that catches SIGTERM and exits successfully,
  # and a kill delay that ensures it gets the chance to see the signal.
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -t 30 -- /bin/bash -c \
    'trap "exit 0" SIGINT SIGTERM SIGALRM; \
     touch marker; \
     sleep 10000' \
    &> $TEST_log &
  local sandbox_pid=$!

  # Synchronize on the child having registered its signal handler.
  until test -f "$SANDBOX_DIR/marker"; do sleep 1; done

  # Send SIGTERM to the sandbox.
  kill -SIGTERM "${sandbox_pid}"

  # The sandbox should exit successfully: if the child says it succeeded, who
  # are we to disagree?
  wait "${sandbox_pid}"
}

function test_basic_timeout() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -T 3 -t 3 -- /bin/bash -c "echo before; sleep 1000; echo after" &> $TEST_log && fail
  expect_log "^before$" ""
}

function test_timeout_exceeded_with_large_kill_delay() {
  # Run the sandbox under a short timeout with a child that catches signals,
  # waits while, then exits with a canned code. Use a kill delay that gives it a
  # chance to do so.
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -T 2 -t 30 -- /bin/bash -c \
    'trap "sleep 1; exit 17" SIGTERM; \
     sleep 10000' \
    &> $TEST_log || code=$?

  # Assuming the trap command was run before the timeout, the "clean shutdown"
  # code should have been allowed to run (returning 17). Otherwise it should
  # have died immediately with SIGTERM.
  local expected=( "17" "143" )
  assert_one_of $expected "$code"
}

function test_timeout_and_kill_delay_exceeded() {
  # Run the sandbox with a child that ignores SIGTERM, and with both a short
  # timeout and a short kill delay.
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -T 2 -t 3 -- /bin/bash -c \
    'trap "" SIGTERM; sleep 1000' &> $TEST_log || code=$?

  # If the trap command was run before the timeout we should have seen SIGKILL
  # (after the kill delay); otherwise SIGTERM should have taken out the child.
  local expected=( "137" "143" )
  assert_one_of $expected "$code"
}

function test_sigint_sends_sigterm() {
  # Run the sandbox with a child whose SIGTERM handler exits with a canned error
  # code.
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -T 100000 -i -- /bin/bash -c \
    'trap "exit 17" SIGTERM; \
     trap "echo should not get here" SIGINT SIGALRM; \
     touch marker; \
     sleep 10000' \
    &> $TEST_log &
  local sandbox_pid=$!

  # Synchronize on the child having registered its signal handler.
  until test -f "$SANDBOX_DIR/marker"; do sleep 1; done

  # Send SIGINT to the sandbox.
  kill -SIGINT "${sandbox_pid}"

  # That should be converted to SIGTERM by the sandbox, causing the child's
  # handler to run.
  wait "${sandbox_pid}" || code=$?
  assert_equals 137 "$code" # SIGNAL_BASE + SIGTERM = 128 + 9
}

function test_debug_logging() {
  touch ${TEST_TMPDIR}/testfile
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -D /tmp/debug -- /bin/true &> $TEST_log || code=$?
  assert_contains "child exited normally with code 0" /tmp/debug
}

function test_debug_logging_does_not_pollute_stdout_nor_stderr() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -D /tmp/debug -l $OUT -L $ERR -- /bin/bash -c "echo out; echo err >&2" &> $TEST_log || code=$?
  assert_equals "out" "$(cat $OUT)"
  assert_equals "err" "$(cat $ERR)"
}

function test_debug_logging_does_not_leak_descriptor() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -D /tmp/debug -- /bin/bash -c "ls /proc/\$\$/fd | tr '\n' ' '; echo ." &> $TEST_log || fail
  expect_log "0 1 2 ."
}

function test_mount_additional_paths_success() {
  mkdir -p ${TEST_TMPDIR}/foo
  mkdir -p ${TEST_TMPDIR}/bar
  touch ${TEST_TMPDIR}/testfile
  mkdir -p ${MOUNT_TARGET_ROOT}/foo
  touch ${MOUNT_TARGET_ROOT}/sandboxed_testfile

  touch /tmp/sandboxed_testfile
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -D /tmp/debug \
    -M ${TEST_TMPDIR}/foo -m ${MOUNT_TARGET_ROOT}/foo \
    -M ${TEST_TMPDIR}/bar \
    -M ${TEST_TMPDIR}/testfile -m ${MOUNT_TARGET_ROOT}/sandboxed_testfile \
    -- /bin/true &> $TEST_log || code=$?
  # mount a directory to a customized path inside the sandbox
  assert_contains "bind mount: ${TEST_TMPDIR}/foo -> ${MOUNT_TARGET_ROOT}/foo\$" /tmp/debug
  # mount a directory to the same path inside the sanxbox
  assert_contains "bind mount: ${TEST_TMPDIR}/bar -> ${TEST_TMPDIR}/bar\$" /tmp/debug
  # mount a file to a customized path inside the sandbox
  assert_contains "bind mount: ${TEST_TMPDIR}/testfile -> ${MOUNT_TARGET_ROOT}/sandboxed_testfile\$" /tmp/debug
  assert_contains "child exited normally with code 0" /tmp/debug
  rm -rf ${MOUNT_TARGET_ROOT}/foo
  rm -rf ${MOUNT_TARGET_ROOT}/sandboxed_testfile
}

function test_mount_additional_paths_relative_path() {
  touch ${TEST_TMPDIR}/testfile
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -D /tmp/debug \
    -M ${TEST_TMPDIR}/testfile -m tmp/sandboxed_testfile \
    -- /bin/true &> $TEST_log || code=$?
  # mount a directory to a customized path inside the sandbox
  expect_log "The -m option must be used with absolute paths only.\$"
}

function test_mount_additional_paths_leading_m() {
  mkdir -p ${TEST_TMPDIR}/foo
  touch ${TEST_TMPDIR}/testfile
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -D /tmp/debug \
    -m /tmp/foo \
    -M ${TEST_TMPDIR}/testfile -m /tmp/sandboxed_testfile \
    -- /bin/true &> $TEST_log || code=$?
  # mount a directory to a customized path inside the sandbox
  expect_log "The -m option must be strictly preceded by an -M option.\$"
}

function test_mount_additional_paths_m_not_preceeded_by_M() {
  mkdir -p ${TEST_TMPDIR}/foo
  mkdir -p ${TEST_TMPDIR}/bar
  touch ${TEST_TMPDIR}/testfile
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -D /tmp/debug \
    -M ${TEST_TMPDIR}/testfile -m /tmp/sandboxed_testfile \
    -m /tmp/foo \
    -M ${TEST_TMPDIR}/bar \
    -- /bin/true &> $TEST_log || code=$?
  # mount a directory to a customized path inside the sandbox
  expect_log "The -m option must be strictly preceded by an -M option.\$"
}

function test_mount_additional_paths_other_flag_between_M_m_pair() {
  mkdir -p ${TEST_TMPDIR}/bar
  touch ${TEST_TMPDIR}/testfile
  $linux_sandbox $SANDBOX_DEFAULT_OPTS \
    -M ${TEST_TMPDIR}/testfile -D /tmp/debug -m /tmp/sandboxed_testfile \
    -M ${TEST_TMPDIR}/bar \
    -- /bin/true &> $TEST_log || code=$?
  # mount a directory to a customized path inside the sandbox
  expect_log "The -m option must be strictly preceded by an -M option.\$"
}

function test_mount_additional_paths_multiple_sources_mount_to_one_target() {
  mkdir -p ${TEST_TMPDIR}/foo
  mkdir -p ${TEST_TMPDIR}/bar
  mkdir -p ${MOUNT_TARGET_ROOT}/foo
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -D /tmp/debug \
    -M ${TEST_TMPDIR}/foo -m ${MOUNT_TARGET_ROOT}/foo \
    -M ${TEST_TMPDIR}/bar -m ${MOUNT_TARGET_ROOT}/foo \
    -- /bin/true &> $TEST_log || code=$?
  # mount a directory to a customized path inside the sandbox
  assert_contains "bind mount: ${TEST_TMPDIR}/foo -> ${MOUNT_TARGET_ROOT}/foo\$" /tmp/debug
  # mount a new source directory to the same target, which will overwrite the previous source path
  assert_contains "bind mount: ${TEST_TMPDIR}/bar -> ${MOUNT_TARGET_ROOT}/foo\$" /tmp/debug
  assert_contains "child exited normally with code 0" /tmp/debug
  rm -rf ${MOUNT_TARGET_ROOT}/foo
}

function test_redirect_output() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -l $OUT -L $ERR -- /bin/bash -c "echo out; echo err >&2" &> $TEST_log || code=$?
  assert_equals "out" "$(cat $OUT)"
  assert_equals "err" "$(cat $ERR)"
}

function test_tmp_is_writable() {
  # If /tmp is not writable on the host, it won't be inside the sandbox.
  test -w /tmp || return 0

  $linux_sandbox $SANDBOX_DEFAULT_OPTS -w /tmp -- /bin/bash -c "rm -f $(mktemp --tmpdir=/tmp)" \
    &> $TEST_log || fail
}

function test_dev_shm_is_writable() {
  # If /dev/shm is not writable on the host, it won't be inside the sandbox.
  test -w /dev/shm || return 0

  # /dev/shm is often a symlink to /run/shm, thus we use readlink to get the canonical path.
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -w "$(readlink -f /dev/shm)" -- /bin/bash -c "rm -f $(mktemp --tmpdir=/dev/shm)" \
    &> $TEST_log || fail
}

function assert_linux_sandbox_exec_time() {
  local user_time_low="$1"; shift
  local user_time_high="$1"; shift
  local sys_time_low="$1"; shift
  local sys_time_high="$1"; shift

  local local_tmp="$(mktemp -d "${OUT_DIR}/assert_linux_sandbox_exec_timeXXXX")"
  local stdout_path="${local_tmp}/stdout"
  local stderr_path="${local_tmp}/stderr"
  local stats_out_path="${local_tmp}/statsfile"
  local stats_out_decoded_path="${local_tmp}/statsfile.decoded"

  cp "${CPU_TIME_SPENDER}" "${SANDBOX_DIR}"
  local cpu_time_spender_sandbox_path="${SANDBOX_DIR}/spend_cpu_time"

  # Sandboxed process will be terminated after 100 seconds if not already dead.
  local code=0
  "${linux_sandbox}" \
      -W "${SANDBOX_DIR}" \
      -T 100 \
      -t 2 \
      -l "${stdout_path}" \
      -L "${stderr_path}" \
      -S "${stats_out_path}" \
      -- \
      "${cpu_time_spender_sandbox_path}" "${user_time_low}" "${sys_time_low}" \
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
  assert_linux_sandbox_exec_time 10 19 0 9
}

function test_stats_high_system_time() {
  assert_linux_sandbox_exec_time 0 9 10 19
}

function test_stats_high_user_time_and_high_system_time() {
  assert_linux_sandbox_exec_time 10 25 10 25
}

function test_sigterm_is_forwarded_to_child() {
  # Run the sandbox with a child that lets us know it saw SIGTERM and then exits
  # gracefully. Use a kill delay that implies it will have a chance to see it.
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -t 30 -- /bin/bash -c \
    'trap "exit 17" SIGTERM; \
     touch marker; \
     sleep 10000' \
    &> $TEST_log &
  local sandbox_pid=$!

  # Synchronize on the child having registered its signal handler.
  until test -f "$SANDBOX_DIR/marker"; do sleep 1; done

  # Send SIGTERM to the sandbox, which should pass it on to the child.
  kill -SIGTERM "${sandbox_pid}"

  # The sandbox should soon exit with the child's exit code.
  wait "${sandbox_pid}" || code=$?
  assert_equals 17 "$code"
}

function test_child_ignores_sigterm_and_sigalrm() {
  # Run the sandbox with a child that ignores SIGTERM (the polite request to
  # shut down that we send below) and SIGALRM (which we use internally and which
  # was previously erroneously forwarded to the child).
  #
  # Set a kill delay of 2 seconds.
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -t 2 -- /bin/bash -c \
    'trap "" SIGTERM SIGALRM; touch marker; sleep 1000' &> $TEST_log &
  local sandbox_pid=$!

  # Synchronize on the child having registered its signal handler.
  until test -f "$SANDBOX_DIR/marker"; do sleep 1; done

  # Send SIGTERM to the sandbox to ask it nicely to shut down.
  kill -SIGTERM "${sandbox_pid}"

  # The sandbox should soon exit with a code that indicates it was due to
  # SIGKILL, from exceeding the kill delay.
  wait "${sandbox_pid}" || code=$?
  assert_equals 137 "$code" # SIGNAL_BASE + SIGTERM = 128 + 9
}

function test_child_ignores_sigterm_and_sigalrm_no_kill_delay() {
  # Run the sandbox with a child that ignores SIGTERM (the polite request to
  # shut down that we send below) and SIGALRM (which we use internally and which
  # was previously erroneously forwarded to the child).
  #
  # Unlike in the test above, don't enable a kill delay.
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -- /bin/bash -c \
    'trap "" SIGTERM SIGALRM; touch marker; sleep 1000' &> $TEST_log &
  local sandbox_pid=$!

  # Synchronize on the child having registered its signal handler.
  until test -f "$SANDBOX_DIR/marker"; do sleep 1; done

  # Send SIGTERM to the sandbox to ask it nicely to shut down.
  kill -SIGTERM "${sandbox_pid}"

  # The sandbox should soon exit with a code that indicates it was due to
  # SIGKILL, from exceeding the kill delay (of zero).
  wait "${sandbox_pid}" || code=$?
  assert_equals 137 "$code" # SIGNAL_BASE + SIGTERM = 128 + 9
}

# Tests that using cgruops v1 with linux_sandbox.cc works, if it's available
function test_cgroups1_memory_limit() {
  if ! grep -E '^cgroup +[^ ]+ +cgroup +.*memory.*' /proc/mounts; then
    echo "No cgroup memory controller mounted, skipping test"
    return 0
  fi
  memmount=$(grep -E '^cgroup +[^ ]+ +cgroup +.*memory.*' /proc/mounts | cut -d' ' -f2)
  if ! grep -E '^[0-9]*:[^:]*memory[^:]*:' /proc/self/cgroup &>/dev/null; then
    echo "Does not use cgroups v1, skipping test"
    return 0
  fi
  memsubdir=$(grep -E '^[0-9]*:[^:]*memory[^:]*:' /proc/self/cgroup | cut -d: -f3)
  memdir="$memmount$memsubdir"
  if [[ ! -w "$memdir" ]]; then
    echo "Cgroups v1 directory not writable, skipping test"
    return 0
  fi
  cat >${TEST_TMPDIR}/run_sandbox_with_cgroups.sh <<EOF
#!/bin/bash
set -euo pipefail
# Runs the sandbox with appropriate cgroups setup
cgroups_self=$memdir
cgroups_parent=\$( dirname "\$cgroups_self" )
cgroups_base=$memdir/blaze_test
mkdir -p "\$cgroups_base" || ( echo "Error creating \$cgroups_base"; exit 1 )
echo \$1 > "\$cgroups_base/memory.limit_in_bytes" \
  || ( echo "Error setting \$1 memory limit"; exit 1 )
cat "\$cgroups_base/memory.limit_in_bytes"
$linux_sandbox $SANDBOX_DEFAULT_OPTS -M "\$cgroups_base" -w "\$cgroups_base" -C "\$cgroups_base" \
  -- /bin/echo hi there
EOF
  chmod +x ${TEST_TMPDIR}/run_sandbox_with_cgroups.sh

  ${TEST_TMPDIR}/run_sandbox_with_cgroups.sh 1000000 &>$TEST_log \
    || fail "Expected sandbox run to succeed"

  expect_log "hi there"

  ${TEST_TMPDIR}/run_sandbox_with_cgroups.sh 1000 &> $TEST_log \
  && fail "Expected sandbox run to fail"

  expect_not_log "hi there"
}

# Tests that using cgruops v2 with linux_sandbox.cc works, if it's available
function test_cgroups2_memory_limit() {
  if ! grep '^0::' /proc/self/cgroup &>/dev/null; then
    echo "Not using cgroups v2, skipping test"
    return 0
  fi
  if ! XDG_RUNTIME_DIR=/run/user/$( id -u ) systemd-run --user --scope true; then
    echo "Not able to use systemd, skipping test"
    return 0
  fi
  cat >${TEST_TMPDIR}/run_sandbox_with_cgroups.sh <<EOF
#!/bin/bash
set -euo pipefail
# Runs the sandbox with appropriate cgroups setup
cgroups_self=/sys/fs/cgroup\$( cut -d: -f3- /proc/self/cgroup)
cgroups_parent=\$( dirname "\$cgroups_self" )
cgroups_base="\$cgroups_parent"/blaze_test
mkdir -p "\$cgroups_base" || ( echo "Error creating \$cgroups_base"; exit 1 )
if ! grep memory "\$cgroups_parent/cgroup.controllers" &>/dev/null ; then
  echo memory >"\$cgroups_parent/cgroup.subtree_control" \
    || ( echo "Error setting subtree control"; exit 1 )
fi
echo \$1 > "\$cgroups_base/memory.max" \
  || ( echo "Error setting \$1 memory limit"; exit 1 )
$linux_sandbox $SANDBOX_DEFAULT_OPTS -C "\$cgroups_base" \
  -- /bin/echo hi there
EOF
  chmod +x ${TEST_TMPDIR}/run_sandbox_with_cgroups.sh

  XDG_RUNTIME_DIR=/run/user/$( id -u ) systemd-run --user --scope \
    ${TEST_TMPDIR}/run_sandbox_with_cgroups.sh 1000000 &>$TEST_log \
    || fail "Expected sandbox run to succeed"

  expect_log "hi there"

  XDG_RUNTIME_DIR=/run/user/$( id -u ) systemd-run --user --scope \
    ${TEST_TMPDIR}/run_sandbox_with_cgroups.sh 1000 &> $TEST_log \
     && fail "Expected sandbox run to fail"

  expect_not_log "hi there"
}

# The test shouldn't fail if the environment doesn't support running it.
[[ "$(uname -s)" = Linux ]] || exit 0
check_sandbox_allowed || exit 0

run_suite "linux-sandbox"
