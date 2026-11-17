#!/usr/bin/env bash
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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# Test that verifies @bazel_tools//tools:bzl_srcs contains all .bzl source
# files underneath @bazel_tools//tools
function test_bzl_srcs() {
  local registered_files=$(bazel query 'kind(source, deps(@bazel_tools//tools:bzl_srcs))')
  registered_files="${registered_files//://}"
  # Find the actual set of .bzl source files. "bazel query @bazel_tools//..." does
  # not work here, because that command currently fails.
  # See https://github.com/bazelbuild/bazel/issues/8859.
  local tools_base="$(bazel info output_base)/external/bazel_tools/tools"
  local tools_base_len=$(echo $tools_base | wc -c)
  local found_files=$(find "$tools_base" -name "*.bzl")

  for found_file in $found_files; do
    expected_label="@bazel_tools//tools/${found_file:tools_base_len}"
    if ! [[ "$registered_files" =~ "$expected_label" ]]; then
      fail "$expected_label was not found under @bazel_tools//tools:bzl_srcs. Found: $registered_files"
    fi
  done
}

function test_toolchain_types_in_rules() {
  local -r toolchain_types=(
    tools/cpp:toolchain_type
    tools/cpp:cc_runtimes_toolchain_type
    tools/cpp:test_runner_toolchain_type
    tools/jdk:toolchain_type
    tools/jdk:runtime_toolchain_type
    tools/jdk:bootstrap_runtime_toolchain_type
    tools/python:toolchain_type
  )
  local -r query="set(${toolchain_types[*]/#/@bazel_tools//})"

  bazel query --output=label_kind "$query" >& "$TEST_log" \
    || fail "Expected query to succeed"
  for toolchain_type in "${toolchain_types[@]}"; do
    expect_log "^toolchain_type rule .*bazel_tools//$toolchain_type\$"
  done

  bazel query --repo_env=BAZEL_INCOMPATIBLE_TOOLCHAIN_TYPES_IN_RULES=1 \
    --output=label_kind "$query" >& "$TEST_log" \
    || fail "Expected query to succeed"
  for toolchain_type in "${toolchain_types[@]}"; do
    expect_log "^alias rule .*bazel_tools//$toolchain_type\$"
  done
}

run_suite "bazel_tools test suite"
