#!/bin/bash
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
# Test that the embedded skylark code is compliant with --all_incompatible_changes.
#
# blaze_enable_disable_tools_defaults_flag_test.sh:
# integration tests for incompatible_disable_tools_defaults_package flag

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/../shell_utils.sh" \
  || { echo "shell_utils.sh not found!" >&2; exit 1; }

#### TESTS #############################################################

function test_enable_defaults_package_option_works() {
  bazel query //tools/defaults:jdk --incompatible_disable_tools_defaults_package=false || fail "Query failed."
}

function test_disable_defaults_package_option_doesnt_works() {
  bazel query //tools/defaults:jdk --incompatible_disable_tools_defaults_package=true && fail "Query expected to fail, but it didn't." || true
}

function test_after_disabling_flag_target_reloaded() {
   mkdir -p a
   cat > a/BUILD <<EOF
filegroup(
  name='a',
)
EOF

   mkdir -p tools/defaults
   cat > tools/defaults/BUILD <<EOF
filegroup(
    name = "jdk",
    srcs = ["//a:a"],
)
EOF
  bazel query 'deps(//tools/defaults:jdk, 1)' >& "$TEST_log" --incompatible_disable_tools_defaults_package=true || fail "Query failed"
  expect_query_targets //tools/defaults:jdk //a:a

  bazel query 'deps(//tools/defaults:jdk, 1)' >& "$TEST_log" --incompatible_disable_tools_defaults_package=false || fail "Query failed"
  expect_query_targets //tools/defaults:jdk @bazel_tools//tools/jdk:{jdk,remote_jdk}

  rm tools/defaults/BUILD
  rm a/BUILD
}

function test_after_enabling_flag_target_reloaded(){
   mkdir -p a
   cat > a/BUILD <<EOF
filegroup(
  name='a',
)
EOF

   mkdir -p tools/defaults
   cat > tools/defaults/BUILD <<EOF
filegroup(
    name = "jdk",
    srcs = ["//a:a"],
)
EOF
  bazel query 'deps(//tools/defaults:jdk, 1)' >& "$TEST_log" --incompatible_disable_tools_defaults_package=false || fail "Query failed"
  expect_query_targets //tools/defaults:jdk @bazel_tools//tools/jdk:{jdk,remote_jdk}


  bazel query 'deps(//tools/defaults:jdk, 1)' >& "$TEST_log" --incompatible_disable_tools_defaults_package=true || fail "Query failed"
  expect_query_targets //tools/defaults:jdk //a:a

  rm tools/defaults/BUILD
  rm a/BUILD
}


function test_independent_target_doesnt_depend_on_flag (){
   mkdir -p a
   cat > a/BUILD <<EOF
genrule(
    name = "genrule",
    outs = ["genrule.txt"],
    cmd = "echo HELLO > \$@",
)
EOF
  bazel build //a:genrule --incompatible_disable_tools_defaults_package=true || fail "Query failed"

  bazel query 'deps(//tools/defaults:jdk, 1)' --incompatible_disable_tools_defaults_package=false || fail "Query failed"

  bazel dump --skyframe=detailed  >& "$TEST_log"

  expect_log_once "^CONFIGURED_TARGET://a:genrule"

  rm a/BUILD
}

run_suite "Bazel incompatible_disable_tools_defaults_package tests"
