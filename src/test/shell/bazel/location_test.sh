#!/usr/bin/env bash
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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_external_location() {
  cat > MODULE.bazel <<EOF
local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(
   name = "bar",
   path = "bar",
)
EOF
  mkdir bar
  touch bar/REPO.bazel
  cat > bar/BUILD <<EOF
genrule(
    name = "baz-rule",
    outs = ["baz"],
    cmd = "echo 'hello' > \"\$@\"",
    visibility = ["//visibility:public"],
)
EOF

  cat > BUILD <<EOF
genrule(
    name = "use-loc",
    srcs = ["@bar//:baz-rule"],
    outs = ["loc"],
    cmd = "cat \$(location @bar//:baz-rule) > \"\$@\"",
)
EOF

  bazel build //:loc &> $TEST_log || fail "Referencing external genrule didn't build"
  assert_contains "hello" bazel-genfiles/loc
}

function test_external_location_tool() {
  cat > MODULE.bazel <<EOF
local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(
   name = "bar",
   path = "bar",
)
EOF
  mkdir bar
  touch bar/REPO.bazel
  cat > bar/BUILD <<EOF
genrule(
    name = "baz-rule",
    outs = ["baz"],
    cmd = "echo '#!/bin/echo hello' > \"\$@\"",
    visibility = ["//visibility:public"],
)
EOF

  cat > BUILD <<EOF
genrule(
    name = "use-loc",
    tools = ["@bar//:baz-rule"],
    outs = ["loc"],
    cmd = "\$(location @bar//:baz-rule) > \"\$@\"",
)
EOF

  bazel build //:loc &> $TEST_log || fail "Referencing external genrule in tools didn't build"
  assert_contains "hello" bazel-genfiles/loc
}

function test_location_trim() {
  mkdir bar
  cat > bar/BUILD <<EOF
genrule(
    name = "baz-rule",
    outs = ["baz"],
    cmd = "echo helloworld > \"\$@\"",
)

genrule(
    name = "loc-rule",
    srcs = [":baz-rule"],
    outs = ["loc"],
    cmd = "echo \$(location  :baz-rule ) > \"\$@\"",
)
EOF

  bazel build //bar:loc || fail "Label was not trimmed before lookup"
}

run_suite "location tests"
