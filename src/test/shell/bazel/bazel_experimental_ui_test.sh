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
# An end-to-end test that Bazel's experimental UI produces reasonable output.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

#### SETUP #############################################################

set -e

add_to_bazelrc "build --genrule_strategy=local"
add_to_bazelrc "test --test_strategy=standalone"

function set_up() {
  mkdir -p pkg
  touch remote_file
  cat > WORKSPACE <<EOF
http_file(name="remote", urls=["file://`pwd`/remote_file"])
EOF
  touch BUILD
}

#### TESTS #############################################################


function test_fetch {
  bazel clean --expunge
  bazel fetch @remote//... --experimental_ui --curses=yes 2>$TEST_log || fail "bazel fetch failed"
  expect_log 'Fetching.*remote_file'
}

run_suite "Bazel-specific integration tests for the experimental UI"
