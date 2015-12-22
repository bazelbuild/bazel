#!/bin/bash
#
# Copyright 2015 The Bazel Authors. All rights reserved.
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
# Test handling of Skylark loads from and in external repositories.
#

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

# The following tests build an instance of a Skylark macro loaded from a
# local_repository, which in turns loads another Skylark file either from
# the external repo or the default repo, depending on the test parameters.
# The tests cover all the valid syntactic variants of the second load. The
# package structure used for the tests is as follows:
#
# ${WORKSPACE_DIR}/
#   WORKSPACE
#   local_pkg/
#     BUILD
#   another_local_pkg/
#     BUILD
#     local_constants.bzl
# ${external_repo}/
#   external_pkg/
#     BUILD
#     macro_def.bzl
#     external_constants.bzl
function run_external_skylark_load_test() {
  load_target_to_test=$1
  expected_test_output=$2

  create_new_workspace
  external_repo=${new_workspace_dir}

  cat > ${WORKSPACE_DIR}/WORKSPACE <<EOF
local_repository(name = "external_repo", path = "${external_repo}")
EOF

  mkdir ${WORKSPACE_DIR}/local_pkg
  cat > ${WORKSPACE_DIR}/local_pkg/BUILD <<EOF
load("@external_repo//external_pkg:macro_def.bzl", "macro")
macro(name="macro_instance")
EOF

  mkdir ${WORKSPACE_DIR}/another_local_pkg
  touch ${WORKSPACE_DIR}/another_local_pkg/BUILD
  cat > ${WORKSPACE_DIR}/another_local_pkg/local_constants.bzl <<EOF
OUTPUT_STRING = "LOCAL!"
EOF

  mkdir ${external_repo}/external_pkg
  touch ${external_repo}/external_pkg/BUILD
  cat > ${external_repo}/external_pkg/macro_def.bzl <<EOF
load("${load_target_to_test}", "OUTPUT_STRING")
def macro(name):
  native.genrule(
      name = name,
      outs = [name + ".txt"],
      cmd = "echo " + OUTPUT_STRING + " > \$@",
  )
EOF

  cat > ${external_repo}/external_pkg/external_constants.bzl <<EOF
OUTPUT_STRING = "EXTERNAL!"
EOF

  cd ${WORKSPACE_DIR}
  bazel build local_pkg:macro_instance >& $TEST_log || \
    fail "Expected build to succeed"
  assert_contains "${expected_test_output}" \
    bazel-genfiles/local_pkg/macro_instance.txt
}

# A label with an explicit external repo reference should be resolved relative
# to the external repo.
function test_load_skylark_from_external_repo_with_pkg_relative_label_load() {
  run_external_skylark_load_test \
    "@external_repo//external_pkg:external_constants.bzl" "EXTERNAL!"
}

# A relative label should be resolved relative to the external package.
function test_load_skylark_from_external_repo_with_pkg_relative_label_load() {
  run_external_skylark_load_test ":external_constants.bzl" "EXTERNAL!"
}

# A relative path should be resolved relative to the current (external)
# directory.
function test_load_skylark_from_external_repo_with_pkg_relative_path_load() {
  run_external_skylark_load_test "external_constants" "EXTERNAL!"
}

# An absolute label with no repo prefix should be resolved relative to the
# current (external) repo.
function test_load_skylark_from_external_repo_with_pkg_relative_path_load() {
  run_external_skylark_load_test "//external_pkg:external_constants.bzl" \
    "EXTERNAL!"
}

# An absolute label with the special "@" prefix should cause be resolved
# relative to the default repo.
function test_load_skylark_from_external_repo_with_repo_relative_label_load() {
  run_external_skylark_load_test "@//another_local_pkg:local_constants.bzl" \
    "LOCAL!"
}

# An absolute path should always be resolved in the default repo.
function test_load_skylark_from_external_repo_with_repo_relative_label_load() {
  run_external_skylark_load_test "/another_local_pkg/local_constants" \
    "LOCAL!"
}

run_suite "Test Skylark loads from/in external repositories"


