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
src_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source ${src_dir}/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }
source ${src_dir}/bazel_sandboxing_test_utils.sh \
  || { echo "bazel_sandboxing_test_utils.sh not found!" >&2; exit 1; }

readonly OUT_DIR="${TEST_TMPDIR}/out"
readonly OUT="${OUT_DIR}/outfile"
readonly ERR="${OUT_DIR}/errfile"
readonly SANDBOX_DIR="${OUT_DIR}/sandbox"

SANDBOX_DEFAULT_OPTS="-S $SANDBOX_DIR"
for dir in /bin* /lib* /usr/bin* /usr/lib*; do
  SANDBOX_DEFAULT_OPTS="$SANDBOX_DEFAULT_OPTS -M $dir"
done

function set_up {
  rm -rf $OUT_DIR
  mkdir -p $SANDBOX_DIR
}

function assert_stdout() {
  assert_equals "$1" "$(cat $OUT)"
}

function assert_output() {
  assert_equals "$1" "$(cat $OUT)"
  assert_equals "$2" "$(cat $ERR)"
}

function test_basic_functionality() {
  $namespace_sandbox $SANDBOX_DEFAULT_OPTS -l $OUT -L $ERR -- /bin/echo hi there || fail
  assert_output "hi there" ""
}

function test_default_user_is_nobody() {
  $namespace_sandbox $SANDBOX_DEFAULT_OPTS -l $OUT -L $ERR -- /usr/bin/id || fail
  assert_output "uid=65534 gid=65534 groups=65534" ""
}

function test_user_switched_to_root() {
  $namespace_sandbox $SANDBOX_DEFAULT_OPTS -r -l $OUT -L $ERR -- /usr/bin/id || fail
  assert_contains "uid=0 gid=0" "$OUT"
}

function test_network_namespace() {
  $namespace_sandbox $SANDBOX_DEFAULT_OPTS -n -l $OUT -L $ERR  -- /bin/ip link ls || fail
  assert_contains "LOOPBACK,UP" "$OUT"
}

function test_ping_loopback() {
  $namespace_sandbox $SANDBOX_DEFAULT_OPTS -n -r -l $OUT -L $ERR  -- /bin/ping -c 1 127.0.0.1 || fail
  assert_contains "1 received" "$OUT"
}

function test_to_stderr() {
  $namespace_sandbox $SANDBOX_DEFAULT_OPTS -l $OUT -L $ERR -- /bin/bash -c "/bin/echo hi there >&2" || fail
  assert_output "" "hi there"
}

function test_exit_code() {
  $namespace_sandbox $SANDBOX_DEFAULT_OPTS -l $OUT -L $ERR -- /bin/bash -c "exit 71" || code=$?
  assert_equals 71 "$code"
}

function test_signal_death() {
  $namespace_sandbox $SANDBOX_DEFAULT_OPTS -l $OUT -L $ERR -- /bin/bash -c 'kill -ABRT $$' || code=$?
  assert_equals 134 "$code" # SIGNAL_BASE + SIGABRT = 128 + 6
}

function test_signal_catcher() {
  $namespace_sandbox $SANDBOX_DEFAULT_OPTS -T 2 -t 3 -l $OUT -L $ERR -- /bin/bash -c \
    'trap "echo later; exit 0" SIGINT SIGTERM SIGALRM; sleep 1000' || code=$?
  assert_equals 142 "$code" # SIGNAL_BASE + SIGALRM = 128 + 14
  assert_stdout "later"
}

function test_basic_timeout() {
  $namespace_sandbox $SANDBOX_DEFAULT_OPTS -T 3 -t 3 -l $OUT -L $ERR -- /bin/bash -c "echo before; sleep 1000; echo after" && fail
  assert_output "before" ""
}

function test_timeout_grace() {
  $namespace_sandbox $SANDBOX_DEFAULT_OPTS -T 2 -t 3 -l $OUT -L $ERR -- /bin/bash -c \
    'trap "echo -n before; sleep 1; echo -n after; exit 0" SIGINT SIGTERM SIGALRM; sleep 1000' || code=$?
  assert_equals 142 "$code" # SIGNAL_BASE + SIGALRM = 128 + 14
  assert_stdout "beforeafter"
}

function test_timeout_kill() {
  $namespace_sandbox $SANDBOX_DEFAULT_OPTS -T 2 -t 3 -l $OUT -L $ERR -- /bin/bash -c \
    'trap "echo before; sleep 1000; echo after; exit 0" SIGINT SIGTERM SIGALRM; sleep 1000' || code=$?
  assert_equals 142 "$code" # SIGNAL_BASE + SIGALRM = 128 + 14
  assert_stdout "before"
}

function test_debug_logging() {
  touch ${TEST_TMPDIR}/testfile
  $namespace_sandbox $SANDBOX_DEFAULT_OPTS -D -M ${TEST_TMPDIR}/testfile -m /tmp/sandboxed_testfile -l $OUT -L $ERR -- /bin/true || code=$?
  assert_contains "mount: /usr/bin\$" "$ERR"
  assert_contains "mount: ${TEST_TMPDIR}/testfile -> <sandbox>/tmp/sandboxed_testfile\$" "$ERR"
}

# The test shouldn't fail if the environment doesn't support running it.
check_supported_platform || exit 0
check_sandbox_allowed || exit 0

run_suite "namespace-runner"
