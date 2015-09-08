#!/bin/bash
#
# Copyright 2015 Google Inc. All rights reserved.
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

readonly WRAPPER="${bazel_data}/src/main/tools/namespace-sandbox"
readonly OUT_DIR="${TEST_TMPDIR}/out"
readonly OUT="${OUT_DIR}/outfile"
readonly ERR="${OUT_DIR}/errfile"
readonly SANDBOX_DIR="${OUT_DIR}/sandbox"

WRAPPER_DEFAULT_OPTS="-S $SANDBOX_DIR"
for dir in /bin* /lib* /usr/bin* /usr/lib*; do
  WRAPPER_DEFAULT_OPTS="$WRAPPER_DEFAULT_OPTS -M $dir"
done

# namespaces which are used by the sandbox were introduced in 3.8, so
# test won't run on earlier kernels
function check_kernel_version {
  if [ "${PLATFORM-}" = "darwin" ]; then
    echo "Test will skip: sandbox is not yet supported on Darwin."
    exit 0
  fi
  MAJOR=$(uname -r | sed 's/^\([0-9]*\)\.\([0-9]*\)\..*/\1/')
  MINOR=$(uname -r | sed 's/^\([0-9]*\)\.\([0-9]*\)\..*/\2/')
  if [ $MAJOR -lt 3 ]; then
    echo "Test will skip: sandbox requires kernel >= 3.8; got $(uname -r)"
    exit 0
  fi
  if [ $MAJOR -eq 3 ] && [ $MINOR -lt 8 ]; then
    echo "Test will skip: sandbox requires kernel >= 3.8; got $(uname -r)"
    exit 0
  fi
}

# Some CI systems might deactivate sandboxing
function check_sandbox_allowed {
  mkdir -p test
  # Create a program that check if unshare(2) is allowed.
  cat <<'EOF' > test/test.c
#define _GNU_SOURCE
#include <sched.h>
int main() {
  return unshare(CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWIPC | CLONE_NEWUSER);
}
EOF
  cat <<'EOF' >test/BUILD
cc_test(name = "sandbox_enabled", srcs = ["test.c"], copts = ["-std=c99"])
EOF
  bazel test //test:sandbox_enabled || {
    echo "Sandboxing disabled, skipping..."
    return false
  }
}

function set_up {
  rm -rf $OUT_DIR
  rm -rf $SANDBOX_DIR

  mkdir -p $OUT_DIR
  mkdir $SANDBOX_DIR
}

function assert_stdout() {
  assert_equals "$1" "$(cat $OUT)"
}

function assert_output() {
  assert_equals "$1" "$(cat $OUT)"
  assert_equals "$2" "$(cat $ERR)"
}

function test_basic_functionality() {
  $WRAPPER $WRAPPER_DEFAULT_OPTS -l $OUT -L $ERR -- /bin/echo hi there || fail
  assert_output "hi there" ""
}

function test_to_stderr() {
  $WRAPPER $WRAPPER_DEFAULT_OPTS -l $OUT -L $ERR -- /bin/bash -c "/bin/echo hi there >&2" || fail
  assert_output "" "hi there"
}

function test_exit_code() {
  $WRAPPER $WRAPPER_DEFAULT_OPTS -l $OUT -L $ERR -- /bin/bash -c "exit 71" || code=$?
  assert_equals 71 "$code"
}

function test_signal_death() {
  $WRAPPER $WRAPPER_DEFAULT_OPTS -l $OUT -L $ERR -- /bin/bash -c 'kill -ABRT $$' || code=$?
  assert_equals 134 "$code" # SIGNAL_BASE + SIGABRT = 128 + 6
}

function test_signal_catcher() {
  $WRAPPER $WRAPPER_DEFAULT_OPTS -T 2 -t 3 -l $OUT -L $ERR -- /bin/bash -c \
    'trap "echo later; exit 0" SIGINT SIGTERM SIGALRM; sleep 1000' || code=$?
  assert_equals 142 "$code" # SIGNAL_BASE + SIGALRM = 128 + 14
  assert_stdout "later"
}

function test_basic_timeout() {
  $WRAPPER $WRAPPER_DEFAULT_OPTS -T 3 -t 3 -l $OUT -L $ERR -- /bin/bash -c "echo before; sleep 1000; echo after" && fail
  assert_output "before" ""
}

function test_timeout_grace() {
  $WRAPPER $WRAPPER_DEFAULT_OPTS -T 2 -t 3 -l $OUT -L $ERR -- /bin/bash -c \
    'trap "echo -n before; sleep 1; echo -n after; exit 0" SIGINT SIGTERM SIGALRM; sleep 1000' || code=$?
  assert_equals 142 "$code" # SIGNAL_BASE + SIGALRM = 128 + 14
  assert_stdout "beforeafter"
}

function test_timeout_kill() {
  $WRAPPER $WRAPPER_DEFAULT_OPTS -T 2 -t 3 -l $OUT -L $ERR -- /bin/bash -c \
    'trap "echo before; sleep 1000; echo after; exit 0" SIGINT SIGTERM SIGALRM; sleep 1000' || code=$?
  assert_equals 142 "$code" # SIGNAL_BASE + SIGALRM = 128 + 14
  assert_stdout "before"
}

function test_debug_logging() {
  touch ${TEST_TMPDIR}/testfile
  $WRAPPER $WRAPPER_DEFAULT_OPTS -D -M ${TEST_TMPDIR}/testfile -m /tmp/sandboxed_testfile -l $OUT -L $ERR -- /bin/true || code=$?
  assert_contains "mount: /usr/bin\$" "$ERR"
  assert_contains "mount: ${TEST_TMPDIR}/testfile -> <sandbox>/tmp/sandboxed_testfile\$" "$ERR"
}

check_kernel_version
check_sandbox_allowed || exit 0
run_suite "namespace-runner"
