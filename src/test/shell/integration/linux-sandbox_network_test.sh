#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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
# Integration tests for the network-dependent aspects of the sandboxing
# spawn strategy. In particular, those tests that specify -N should be in
# this file, but general tests can be kept in the more general
# linux-sandbox_test.sh.
#

set -euo pipefail

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/../sandboxing_test_utils.sh" \
  || { echo "sandboxing_test_utils.sh not found!" >&2; exit 1; }

readonly OUT_DIR="${TEST_TMPDIR}/out"
readonly SANDBOX_DIR="${OUT_DIR}/sandbox"

SANDBOX_DEFAULT_OPTS="-W $SANDBOX_DIR"

function set_up {
  rm -rf $OUT_DIR
  mkdir -p $SANDBOX_DIR
}

function test_network_no_namespace() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS  -- ip link ls &> $TEST_log || fail
  expect_log "LOOPBACK.*UP"
}

function test_network_namespace() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -N  -- ip link ls &> $TEST_log || fail
  expect_log "LOOPBACK,UP"
}

function test_ping_loopback() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -N -R -- \
    /bin/sh -c 'ping6 -c 1 ::1 || ping -c 1 127.0.0.1' &>$TEST_log || fail
  expect_log "1 received"
}

function test_ping_no_loopback() {
  $linux_sandbox $SANDBOX_DEFAULT_OPTS -n -R -- \
    /bin/sh -c 'ping6 -c 1 ::1 || ping -c 1 127.0.0.1' &>$TEST_log && fail
  expect_not_log "LOOPBACK.*UP"
}

# The test shouldn't fail if the environment doesn't support running it.
[[ "$(uname -s)" = Linux ]] || exit 0
check_sandbox_allowed || exit 0

run_suite "linux-sandbox-network"
