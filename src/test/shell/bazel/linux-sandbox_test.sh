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
source "${CURRENT_DIR}/bazel_sandboxing_test_utils.sh" \
  || { echo "bazel_sandboxing_test_utils.sh not found!" >&2; exit 1; }

readonly OUT_DIR="${TEST_TMPDIR}/out"
readonly OUT="${OUT_DIR}/outfile"
readonly ERR="${OUT_DIR}/errfile"
readonly SANDBOX_DIR="${OUT_DIR}/sandbox"
readonly MOUNT_TARGET_ROOT="${TEST_SRCDIR}/targets"

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
  expect_log "$(id)"
}

function test_user_switched_to_root() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -R -- /usr/bin/id &> $TEST_log || fail
  expect_log "uid=0(root) gid=0(root) groups=0(root)"
}

function test_user_switched_to_nobody() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -U -- /usr/bin/id &> $TEST_log || fail
  expect_log "uid=65534(nobody) gid=65534(nogroup) groups=65534(nogroup)"
}

function test_network_namespace() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -N  -- /bin/ip link ls &> $TEST_log || fail
  expect_log "LOOPBACK,UP"
}

function test_ping_loopback() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -N -R -- \
    /bin/sh -c 'ping6 -c 1 ::1 || ping -c 1 127.0.0.1' &>$TEST_log || fail
  expect_log "1 received"
}

function test_exit_code() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -- /bin/bash -c "exit 71" &> $TEST_log || code=$?
  assert_equals 71 "$code"
}

function test_signal_death() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -- /bin/bash -c 'kill -ABRT $$' &> $TEST_log || code=$?
  assert_equals 134 "$code" # SIGNAL_BASE + SIGABRT = 128 + 6
}

# Tests that even when the child catches SIGTERM and exits with code 0, that the sandbox exits with
# code 142 (telling us about the expired timeout).
function test_signal_catcher() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -T 2 -t 3 -- /bin/bash -c \
    'trap "echo later; exit 0" SIGINT SIGTERM SIGALRM; sleep 1000' &> $TEST_log || code=$?
  assert_equals 142 "$code" # SIGNAL_BASE + SIGALRM = 128 + 14
  expect_log "^later$"
}

function test_basic_timeout() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -T 3 -t 3 -- /bin/bash -c "echo before; sleep 1000; echo after" &> $TEST_log && fail
  expect_log "^before$" ""
}

function test_timeout_grace() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -T 2 -t 3 -- /bin/bash -c \
    'trap "echo -n before; sleep 1; echo -n after; exit 0" SIGINT SIGTERM SIGALRM; sleep 1000' &> $TEST_log || code=$?
  assert_equals 142 "$code" # SIGNAL_BASE + SIGALRM = 128 + 14
  expect_log "^beforeafter$"
}

function test_timeout_kill() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -T 2 -t 3 -- /bin/bash -c \
    'trap "echo before; sleep 1000; echo after; exit 0" SIGINT SIGTERM SIGALRM; sleep 1000' &> $TEST_log || code=$?
  assert_equals 142 "$code" # SIGNAL_BASE + SIGALRM = 128 + 14
  expect_log "^before$"
}

function test_debug_logging() {
  touch ${TEST_TMPDIR}/testfile
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -D -- /bin/true &> $TEST_log || code=$?
  expect_log "child exited normally with exitcode 0"
}

function test_mount_additional_paths_success() {
  mkdir -p ${TEST_TMPDIR}/foo
  mkdir -p ${TEST_TMPDIR}/bar
  touch ${TEST_TMPDIR}/testfile
  mkdir -p ${MOUNT_TARGET_ROOT}/foo
  touch ${MOUNT_TARGET_ROOT}/sandboxed_testfile

  touch /tmp/sandboxed_testfile
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -D \
    -M ${TEST_TMPDIR}/foo -m ${MOUNT_TARGET_ROOT}/foo \
    -M ${TEST_TMPDIR}/bar \
    -M ${TEST_TMPDIR}/testfile -m ${MOUNT_TARGET_ROOT}/sandboxed_testfile \
    -- /bin/true &> $TEST_log || code=$?
  # mount a directory to a customized path inside the sandbox
  expect_log "bind mount: ${TEST_TMPDIR}/foo -> ${MOUNT_TARGET_ROOT}/foo\$"
  # mount a directory to the same path inside the sanxbox
  expect_log "bind mount: ${TEST_TMPDIR}/bar -> ${TEST_TMPDIR}/bar\$"
  # mount a file to a customized path inside the sandbox
  expect_log "bind mount: ${TEST_TMPDIR}/testfile -> ${MOUNT_TARGET_ROOT}/sandboxed_testfile\$"
  expect_log "child exited normally with exitcode 0"
  rm -rf ${MOUNT_TARGET_ROOT}/foo
  rm -rf ${MOUNT_TARGET_ROOT}/sandboxed_testfile
}

function test_mount_additional_paths_relative_path() {
  touch ${TEST_TMPDIR}/testfile
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -D \
    -M ${TEST_TMPDIR}/testfile -m tmp/sandboxed_testfile \
    -- /bin/true &> $TEST_log || code=$?
  # mount a directory to a customized path inside the sandbox
  expect_log "The -m option must be used with absolute paths only.\$"
}

function test_mount_additional_paths_leading_m() {
  mkdir -p ${TEST_TMPDIR}/foo
  touch ${TEST_TMPDIR}/testfile
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -D \
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
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -D \
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
    -M ${TEST_TMPDIR}/testfile -D -m /tmp/sandboxed_testfile \
    -M ${TEST_TMPDIR}/bar \
    -- /bin/true &> $TEST_log || code=$?
  # mount a directory to a customized path inside the sandbox
  expect_log "The -m option must be strictly preceded by an -M option.\$"
}

function test_mount_additional_paths_multiple_sources_mount_to_one_target() {
  mkdir -p ${TEST_TMPDIR}/foo
  mkdir -p ${TEST_TMPDIR}/bar
  mkdir -p ${MOUNT_TARGET_ROOT}/foo
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -D \
    -M ${TEST_TMPDIR}/foo -m ${MOUNT_TARGET_ROOT}/foo \
    -M ${TEST_TMPDIR}/bar -m ${MOUNT_TARGET_ROOT}/foo \
    -- /bin/true &> $TEST_log || code=$?
  # mount a directory to a customized path inside the sandbox
  expect_log "bind mount: ${TEST_TMPDIR}/foo -> ${MOUNT_TARGET_ROOT}/foo\$"
  # mount a new source directory to the same target, which will overwrite the previous source path
  expect_log "bind mount: ${TEST_TMPDIR}/bar -> ${MOUNT_TARGET_ROOT}/foo\$"
  expect_log "child exited normally with exitcode 0"
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

# The test shouldn't fail if the environment doesn't support running it.
check_supported_platform || exit 0
check_sandbox_allowed || exit 0

run_suite "linux-sandbox"
