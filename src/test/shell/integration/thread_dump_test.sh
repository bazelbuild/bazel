#!/usr/bin/env bash
#
# Copyright 2025 The Bazel Authors. All rights reserved.
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

set -eu

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  # Create a new workspace to ensure the output_base is clean
  create_new_workspace
}

# Regression test for b/449656472.
# Test that the thread dump works with build command after a non-build command.
function test_info_and_build() {
  cat > BUILD <<'EOF'
genrule(
  name = "gen",
  outs = ["out"],
  cmd = "touch $@",
)
EOF

  bazel info --experimental_enable_thread_dump --experimental_thread_dump_interval=1s &> "${TEST_log}" || fail "Expected info to succeed"
  bazel build --experimental_enable_thread_dump --experimental_thread_dump_interval=1s //:gen &> "${TEST_log}" || fail "Expected build to succeed"
}

run_suite "thread dump tests"