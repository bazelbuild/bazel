#!/usr/bin/env bash
#
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
# Test lightweight spawn stats generation in Bazel
#

set -eu

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  cat > BUILD <<EOF
genrule(
    name = "foo",
    cmd = "echo hello > \$@",
    outs = ["foo.txt"],
)
EOF
}

function test_order() {
  # Ensure the new stats are printed before Build completed
  bazel build :foo 2>&1 | tee ${TEST_log} | sed -n '/process/,$p' | grep "Build complete" || fail "Expected \"process\" to be followed by \"Build completed\""
}

# Single execution of Bazel
function statistics_single() {
  flags=$1 # flags to pass to Bazel
  expect=$2 # string to expect

  echo "Starting single run for $flags $expect" &> $TEST_log
  output=`bazel build :foo $flags 2>&1 | tee ${TEST_log} | grep " process" | tr -d '\r'`

  if ! [[ $output =~ ${expect} ]]; then
    fail "bazel ${flags}: Want |${expect}|, got |${output}| "
  fi

  echo "Done $flags $expect" &> $TEST_log
}

function test_local() {
  statistics_single "--spawn_strategy=local" ", 1 local"
}

function test_local_sandbox() {
  if [[ "$PLATFORM" == "linux" ]]; then
    statistics_single "--spawn_strategy=linux-sandbox" ", 1 linux-sandbox"
  fi
}

# We are correctly resetting the counts
function test_repeat() {
  flags="--spawn_strategy=local"
  statistics_single $flags ", 1 local"
  bazel clean $flags
  statistics_single $flags ", 1 local"
}

# Locally cached results are not yet displayed
function test_localcache() {
  flags="--spawn_strategy=local"
  # We are correctly resetting the counts
  statistics_single $flags ", 1 local"
  statistics_single $flags "1 process: 1 internal."
}

run_suite "bazel statistics tests"


