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
EOF
}

#### TESTS #############################################################

function test_basic_progress() {
  bazel test --curses=yes --color=yes pkg:true 2>$TEST_log || fail "bazel test failed"
  # some progress indicator is shown
  expect_log '\[[0-9,]* / [0-9,]*\]'
  # something is written in green
  expect_log $'\x1b\[32m'
  # curses are used to delete at least one line
  expect_log $'\x1b\[1A\x1b\[K'
}

function test_line_wrapping() {
  bazel test --curses=yes --color=yes --terminal_columns=5 pkg:true 2>$TEST_log || fail "bazel test failed"
  # curses are used to delete at least one line
  expect_log $'\x1b\[1A\x1b\[K'
  # something is written in green
  expect_log $'\x1b\[32m'
  # lines are wrapped, hence at least one line should end with backslash
  expect_log '\\'$'\r''$'
}

function test_noline_wrapping_color_nocurses() {
  bazel test --curses=no --color=yes --terminal_columns=5 pkg:true 2>$TEST_log || fail "bazel test failed"
  # something is written in green
  expect_log $'\x1b\[32m'
  # no lines are deleted
  expect_not_log $'\x1b\[K'
  # as no line wrapping occurs, no backlsash should be before a carriage return
  expect_not_log '\\'$'\r'
}


run_suite "Basic integration tests for the standard UI"
