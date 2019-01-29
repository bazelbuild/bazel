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
# Test the local disk cache
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_local_action_cache() {
  local cache="${TEST_TMPDIR}/cache"
  local execution_file="${TEST_TMPDIR}/run.log"
  local input_file="foo.in"
  local output_file="bazel-genfiles/foo.txt"
  local flags="--disk_cache=$cache"

  rm -rf $cache
  mkdir $cache

  touch WORKSPACE
  # No sandboxing, side effect is needed to detect action execution
  cat > BUILD <<EOF
genrule(
    name = "foo",
    cmd = "echo run > $execution_file && cat \$< >\$@",
    srcs = ["$input_file"],
    outs = ["foo.txt"],
    tags = ["no-sandbox"],
)
EOF

  # CAS is empty, cache miss
  echo 0 >"${execution_file}"
  echo 1 >"${input_file}"
  bazel build $flags :foo &> $TEST_log || fail "Build failed"
  assert_equals "1" $(cat "${output_file}")
  assert_equals "run" $(cat "${execution_file}")

  # CAS doesn't have output for this input, cache miss
  echo 0 >"${execution_file}"
  echo 2 >"${input_file}"
  bazel build $flags :foo &> $TEST_log || fail "Build failed"
  assert_equals "2" $(cat "${output_file}")
  assert_equals "run" $(cat "${execution_file}")

  # Cache hit, no action run/no side effect
  echo 0 >"${execution_file}"
  echo 1 >"${input_file}"
  bazel build $flags :foo &> $TEST_log || fail "Build failed"
  assert_equals "1" $(cat "${output_file}")
  assert_equals "0" $(cat "${execution_file}")
}

run_suite "local action cache test"
