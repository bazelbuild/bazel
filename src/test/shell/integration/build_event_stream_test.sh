#!/bin/bash
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
# An end-to-end test that Bazel's experimental UI produces reasonable output.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

#### SETUP #############################################################

set -e

function set_up() {
  mkdir -p pkg
  cat > pkg/true.sh <<EOF
#!/bin/sh
exit 0
EOF
  chmod 755 pkg/true.sh
  cat > pkg/BUILD <<EOF
sh_test(
  name = "true",
  srcs = ["true.sh"],
)
test_suite(
  name = "suite",
  tests = ["true"],
)
EOF
}

#### TESTS #############################################################

function test_basic() {
  # Basic properties of the event stream
  # - a completed target explicity requested should be reported
  # - after success the stream should close naturally, without any
  #   reports about aborted events.
  bazel test --experimental_build_event_text_file=$TEST_log pkg:true \
    || fail "bazel test failed"
  expect_log 'pkg:true'
  expect_not_log 'aborted'
}

function test_suite() {
  # ...same true when running a test suite containing that test
  bazel test --experimental_build_event_text_file=$TEST_log pkg:suite \
    || fail "bazel test failed"
  expect_log 'pkg:true'
  expect_not_log 'aborted'
}

function test_test_summary() {
  # Requesting a test, we expect
  # - precisely one test summary (for the single test we run)
  # - that is properly chained (no additional progress events)
  bazel test --experimental_build_event_text_file=$TEST_log pkg:true \
    || fail "bazel test failed"
  expect_log_once '^test_summary '
  expect_log_once '^progress '
  expect_not_log 'aborted'
}

function test_multiple_transports() {
  # Verifies usage of multiple build event transports at the same time
    bazel test \
      --experimental_build_event_text_file=test_multiple_transports.txt \
      --experimental_build_event_binary_file=test_multiple_transports.bin \
      pkg:suite || fail "bazel test failed"
  [ -f test_multiple_transports.txt ] || fail "Missing expected file test_multiple_transports.txt"
  [ -f test_multiple_transports.bin ] || fail "Missing expected file test_multiple_transports.bin"
}

run_suite "Integration tests for the build event stream"
