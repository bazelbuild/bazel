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
# An end-to-end test for Bazel's option handling

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

#### SETUP #############################################################

set -e

function set_up() {
  mkdir -p pkg
  # have test with a long name, to be able to test line breaking in the output
  cat > pkg/xxxxxxxxxxxxxxxxxxxxxxxxxtrue.sh <<EOF
#!/bin/sh
exit 0
EOF
  chmod 755 pkg/xxxxxxxxxxxxxxxxxxxxxxxxxtrue.sh
  cat > pkg/BUILD <<EOF
sh_test(
  name = "xxxxxxxxxxxxxxxxxxxxxxxxxtrue",
  srcs = ["xxxxxxxxxxxxxxxxxxxxxxxxxtrue.sh"],
)
EOF
}

#### TESTS #############################################################

function test_terminal_columns_honored() {
  setup_bazelrc
  cat >>$TEST_TMPDIR/bazelrc <<EOF
build --terminal_columns=6
EOF
  bazel test --curses=yes --color=yes pkg:xxxxxxxxxxxxxxxxxxxxxxxxxtrue \
      2>$TEST_log || fail "bazel test failed"
  # the lines are wrapped to 6 characters
  expect_log '^xxxx'
  expect_not_log '^xxxxxxx'
}

function test_options_override() {
  setup_bazelrc
  cat >>$TEST_TMPDIR/bazelrc <<EOF
build --terminal_columns=6
EOF
  bazel test --curses=yes --color=yes --terminal_columns=10 \
      pkg:xxxxxxxxxxxxxxxxxxxxxxxxxtrue 2>$TEST_log || fail "bazel test failed"
  # the lines are wrapped to 10 characters
  expect_log '^xxxxxxxx'
  expect_not_log '^xxxxxxxxxxx'
}

run_suite "Integration tests for rc options handling"
