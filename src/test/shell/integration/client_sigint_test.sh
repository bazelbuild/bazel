#!/usr/bin/env bash
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
#
# Test of client/server SIGINT handling.  This test proves that the server
# gracefully handles SIGINT.

NO_SIGNAL_OVERRIDE=1
# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# This function doesn't work well with the sandbox because testfifo and
# sleepyfifo are created outside the genrule.
# TODO(bazel-team): Make this test compliant with the behavior of the sandbox
function runbazel() {
  startup_opt=$1; shift

  local sleepyfifo=x/sleepyfifo
  local testfifo=x/testfifo
  mkdir -p x || fail "Can't create x"
  cat > x/BUILD << EOF
genrule(
  name = "sleepy",
  srcs = [],
  outs = ["sleepy.out"],
  local = 1,
  cmd = "echo 'hi test' > $testfifo; cat $sleepyfifo; sleep 9999"
)
EOF

  mkfifo $testfifo $sleepyfifo || fail "Couldn't create FIFOs under x"

  set -m
  bazel $startup_opt build --experimental_ui_debug_all_events \
      --package_path . //x:sleepy >& $TEST_log &
  local pid=$!

  echo "${PRODUCT_NAME} running in background with pid $pid"
  local testfifocontents=$(cat $testfifo)
  echo "hi sleepy" > $sleepyfifo
  echo "Interrupting pid $pid"
  kill -INT $pid; sleep 3

  status=0
  # We expect the wait instruction to fail given that the build is interrupted.
  wait $pid || status=$?
  assert_equals 8 $status # Interruption exit code
  assert_equals "hi test" "$testfifocontents"
  set +m
}

function tear_down() {
  bazel shutdown
  rm -rf x
}

function assert_sigint_stops_build() {
  runbazel $1

  # Must have loaded package 'x':
  expect_log 'Loading package: x'
  expect_log 'Elapsed time'
  expect_log 'build interrupted'
}

function test_sigint_server_mode() {
  assert_sigint_stops_build "--nobatch"
}

function test_sigint_batch_mode() {
  assert_sigint_stops_build "--batch"
}

run_suite "Tests of SIGINT on ${PRODUCT_NAME}"
