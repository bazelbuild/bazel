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
# Tests for package labels that cross repository boundaries
# Includes regression tests for #1592.
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# Test cases are mostly targeting WORKSPACE
add_to_bazelrc "common --enable_workspace"

function write_rule {
  name=$1
  shift
  srcs=""
  for src in "$@"; do
    srcs="\"$src\", $srcs"
  done

  cat <<EOF
genrule(
  name = "$name",
  srcs = [
    $srcs
  ],
  outs = ["$name.out"],
  cmd = "echo $name > \$@",
  visibility = ["//visibility:public"],
)
EOF
}

# Basic use case: Define a repository nested in the main repository, check
# correct targets are visible
function test_basic_cross_repo_targets {
  cat > WORKSPACE <<EOF
local_repository(
    name = "bar",
    path = "bar",
)
EOF

  write_rule depend_via_repo "@bar//:bar" >> BUILD
  write_rule depend_via_repo_subdir "@bar//subbar:subbar" >> BUILD
  write_rule depend_directly "//bar:bar" >> BUILD
  write_rule depend_directly_subdir "//bar/subbar:subbar" >> BUILD

  mkdir -p bar
  touch bar/WORKSPACE
  write_rule bar >> bar/BUILD
  mkdir -p bar/subbar
  write_rule subbar >> bar/subbar/BUILD

  # These should succeed, they use the correct label.
  bazel build //:depend_via_repo || fail "build should succeed"
  bazel build //:depend_via_repo_subdir || fail "build should succeed"

  # These should fail, they is using the wrong label that crosses the
  # repository boundary.
  bazel build //:depend_directly >& $TEST_log && fail "build should fail"
  expect_log "no such package 'bar': Invalid package reference bar crosses into repository @@bar"
  bazel build //:depend_directly_subdir >& $TEST_log && fail "build should fail"
  expect_log "no such package 'bar/subbar': Invalid package reference bar/subbar crosses into repository @@bar"
}

function test_top_level_local_repository {
  cat > WORKSPACE <<EOF
local_repository(
    name = "loc",
    path = ".",
)
EOF

  write_rule foo >> BUILD

  # These should succeed, they use the correct label.
  bazel build //:foo || fail "build should succeed"
  bazel build @loc//:foo || fail "build should succeed"
}

# Loading rules.
function test_workspace_loads_rules {
  cat > WORKSPACE <<EOF
load("//baz:rules.bzl", "baz_rule")
local_repository(
    name = "bar",
    path = "bar",
)
EOF

  write_rule depend_via_repo "@bar//:bar" >> BUILD
  write_rule depend_directly "//bar:bar" >> BUILD

  mkdir -p bar
  touch bar/WORKSPACE
  write_rule bar >> bar/BUILD

  mkdir -p baz
  touch baz/BUILD
  cat > baz/rules.bzl <<EOF
baz_rule = 'baz_rule'
EOF

  # This should succeed, it uses the correct label.
  bazel build //:depend_via_repo || fail "build should succeed"

  # This should fail, it is using the wrong label that crosses the repository
  # boundary.
  bazel build //:depend_directly >& $TEST_log && fail "build should fail"
  expect_log "no such package 'bar': Invalid package reference bar crosses into repository @@bar"
}

# Load rules via an invalid label.
function test_workspace_loads_rules_failure {
  cat > WORKSPACE <<EOF
load("//bar:rules.bzl", "bar_rule")
local_repository(
    name = "bar",
    path = "bar",
)
EOF

  write_rule depend_via_repo "@bar//:bar" >> BUILD

  mkdir -p bar
  touch bar/WORKSPACE
  write_rule bar >> bar/BUILD
  cat > bar/rules.bzl <<EOF
bar_rule = 'bar_rule'
EOF

  # This should fail in workspace parsing.
  bazel build //:depend_via_repo >& $TEST_log && fail "build should fail"
  # TODO(jcater): Show a better error when this occurs
  #expect_log "no such package 'bar': Package crosses into repository @bar"
  # This is the current error shown.
  expect_log "cycles detected"
}

function test_top_level_local_repository {
  cat > WORKSPACE <<EOF
local_repository(
    name = "loc",
    path = ".",
)
EOF

  write_rule foo >> BUILD

  # These should succeed, they use the correct label.
  bazel build //:foo || fail "build should succeed"
  bazel build @loc//:foo || fail "build should succeed"
}

function test_incremental_add_repository {
  # Empty workspace, defines no local_repository.
  cat > WORKSPACE

  mkdir -p bar
  write_rule bar >> bar/BUILD
  mkdir -p bar/subbar
  write_rule subbar >> bar/subbar/BUILD

  bazel query //bar/... >& $TEST_log || fail "query should succeed"

  # Now add the local_repository and WORKSPACE.
  cat > WORKSPACE <<EOF
local_repository(
    name = "bar",
    path = "bar",
)
EOF
  touch bar/WORKSPACE

  # These should now fail, using the incorrect label.
  bazel query //bar:bar >& $TEST_log && fail "build should fail"
  expect_log "no such package 'bar': Invalid package reference bar crosses into repository @@bar"
  bazel query //bar/subbar:subbar >& $TEST_log && fail "build should fail"
  expect_log "no such package 'bar/subbar': Invalid package reference bar/subbar crosses into repository @@bar"

  # These should succeed.
  echo "about to test @bar//"
  bazel query @bar//:bar || fail "query should succeed"
  bazel query @bar//subbar:subbar || fail "query should succeed"
}

function test_incremental_remove_repository {
  # Add the local_repository and WORKSPACE.
  cat > WORKSPACE <<EOF
local_repository(
    name = "bar",
    path = "bar",
)
EOF
  mkdir -p bar
  touch bar/WORKSPACE
  write_rule bar >> bar/BUILD
  mkdir -p bar/subbar
  write_rule subbar >> bar/subbar/BUILD

  # These should succeed.
  echo "about to test @bar//"
  bazel query @bar//:bar || fail "query should succeed"
  bazel query @bar//subbar:subbar || fail "query should succeed"

  # Now remove the workspace and the local_repository.
  cat > WORKSPACE
  rm bar/WORKSPACE

  # These should now succeed.
  bazel query //bar:bar || fail "query should succeed"
  bazel query //bar/subbar:subbar || fail "query should succeed"


  # These should now fail.
  bazel query @bar//:bar >& $TEST_log && fail "build should fail"
  expect_log "No repository visible as '@bar' from main repository"
  bazel query @bar//subbar:subbar >& $TEST_log && fail "build should fail"
  expect_log "No repository visible as '@bar' from main repository"
}

# Test for https://github.com/bazelbuild/bazel/issues/2580
# This issue does not involve a local repository but it is triggered by the
# local repository cross-reference check.
function test_workspace_directory {
  cat > WORKSPACE <<EOF
load('//pkg/WORKSPACE:ext.bzl', 'VALUE')
EOF

  mkdir -p pkg/WORKSPACE

  cat > pkg/WORKSPACE/BUILD <<EOF
exports_files(['ext.bzl'])
EOF

  cat > pkg/WORKSPACE/ext.bzl <<EOF
VALUE = 'a value'
EOF

  # These should succeed, they use the correct label.
  bazel build //pkg/WORKSPACE:all || fail "build should succeed"
}

# TODO(katre): Add tests to verify incremental package reloads are necessary and correct.
# - /WORKSPACE edited, rule not changed - no reload
# - /WORKSPACE not edited, /dir/WORKSPACE added or removed - only packages in
#   /dir invalidated and re-loaded

run_suite "cross-repository tests"
