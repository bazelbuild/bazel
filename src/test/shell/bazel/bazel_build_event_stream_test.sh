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
# An end-to-end test for bazel-specific parts of the build-event stream.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

#### SETUP #############################################################

set -e

function set_up() {
  mkdir -p pkg
  touch pkg/BUILD
}

#### TESTS #############################################################


function test_fetch_test() {
  # We expect the "fetch" command to generate at least a minimally useful
  # build-event stream.
  bazel shutdown
  rm -f "${TEST_log}"
  bazel fetch --build_event_text_file="${TEST_log}" //pkg/... \
      || fail "bazel fetch failed"
  [ -f "${TEST_log}" ] \
      || fail "fetch did not generate requested build-event file"
  expect_log '^started'
  expect_log '^finished'
  expect_log 'name: "SUCCESS"'
}

run_suite "Bazel-specific integration tests for the build-event stream"
