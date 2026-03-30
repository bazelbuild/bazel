#!/usr/bin/env bash
#
# Copyright 2024 The Bazel Authors. All rights reserved.
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

function test_profiler_disabled() {
  cat > BUILD <<'EOF'
genrule(
  name = "gen",
  outs = ["out"],
  cmd = "touch $@",
)
EOF

  bazel build //:gen || fail "Expected build to succeed"

  output_base=$(bazel info output_base)
  if [[ $(find "$output_base" -maxdepth 1 -name "*.jfr") ]]; then
    fail "Expected no profiler outputs"
  fi
}

function do_test_profiler_enabled() {
  local -r type="$1"

  cat > BUILD <<'EOF'
genrule(
  name = "gen",
  outs = ["out"],
  cmd = "touch $@",
)
EOF

 bazel build --experimental_command_profile="${type}" //:gen \
     || fail "Expected build to succeed"

 if ! [[ -f "$(bazel info output_base)/${type}.jfr" ]]; then
     fail "Expected profiler output"
 fi
}

function test_cpu_profiler_enabled() {
  do_test_profiler_enabled cpu
}

function test_wall_profiler_enabled() {
  do_test_profiler_enabled wall
}

function test_alloc_profiler_enabled() {
  do_test_profiler_enabled alloc
}

function test_lock_profiler_enabled() {
  do_test_profiler_enabled lock
}

run_suite "command profiler tests"
