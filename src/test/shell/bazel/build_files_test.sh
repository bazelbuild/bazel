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
#
# Tests the proper checking of BUILD and BUILD.bazel files.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

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

# Only a BUILD file is present: sees rules in BUILD.
function test_build_only {
  create_new_workspace
  write_rule build_only >> BUILD

  bazel build //:build_only >& $TEST_log || fail "build should succeed"
}

# Only a BUILD.bazel file is present: sees rules in BUILD.bazel.
function test_build_bazel_only {
  create_new_workspace
  write_rule build_bazel_only >> BUILD.bazel

  bazel build //:build_bazel_only >& $TEST_log || fail "build should succeed"
}

# BUILD and BUILD.bazel file is present: sees rules in BUILD.bazel.
function test_build_and_build_bazel {
  create_new_workspace
  write_rule build_only >> BUILD
  write_rule build_bazel_only >> BUILD.bazel

  bazel build //:build_bazel_only >& $TEST_log || fail "build should succeed"
  # This rule doesn't actually exist.
  bazel build //:build_only >& $TEST_log && fail "build shouldn't succeed"
  expect_log "no such target '//:build_only'"
}

function test_multiple_package_roots {
  # Create a main workspace with a BUILD.bazel file.
  create_new_workspace
  write_rule build_bazel_only > BUILD.bazel

  # Create an alternate package path with a BUILD file.
  local other_root=$TEST_TMPDIR/other_root/${WORKSPACE_NAME}
  mkdir -p $other_root
  write_rule build_only > $other_root/BUILD

  add_to_bazelrc "build --package_path $other_root:."
  bazel build //:build_only >& $TEST_log || fail "build should succeed"
  # This rule doesn't actually exist.
  bazel build //:build_bazel_only >& $TEST_log && fail "build shouldn't succeed"
  expect_log "no such target '//:build_bazel_only'"
}

function test_build_as_target() {
  local -r pkg="${FUNCNAME}"
  mkdir -p "$pkg/BUILD" || fail "could not create \"$pkg/BUILD\""
  echo 'filegroup(name = "BUILD", srcs = [])' > "$pkg/BUILD/BUILD.bazel" || fail
  # Note the "shorthand" $pkg/BUILD syntax, not $pkg/BUILD:BUILD.
  bazel build "$pkg/BUILD" || fail "Expected success"
}

run_suite "build files tests"

