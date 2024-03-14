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
# Test handling of Starlark loads from and in external repositories.
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# The following tests build an instance of a Starlark macro loaded from a
# local_repository, which in turns loads another Starlark file either from
# the external repo or the main repo, depending on the test parameters.
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
function run_external_starlark_load_test() {
  load_target_to_test=$1
  expected_test_output=$2

  create_new_workspace
  external_repo=${new_workspace_dir}

  cat > ${WORKSPACE_DIR}/WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
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
function test_load_starlark_from_external_repo_with_pkg_relative_label_load() {
  run_external_starlark_load_test \
    "@external_repo//external_pkg:external_constants.bzl" "EXTERNAL!"
}

# A relative label should be resolved relative to the external package.
function test_load_starlark_from_external_repo_with_pkg_relative_label_load() {
  run_external_starlark_load_test ":external_constants.bzl" "EXTERNAL!"
}

# An absolute label with no repo prefix should be resolved relative to the
# current (external) repo.
function test_load_starlark_from_external_repo_with_pkg_relative_path_load() {
  run_external_starlark_load_test "//external_pkg:external_constants.bzl" \
    "EXTERNAL!"
}

# An absolute label with the special "@" prefix should cause be resolved
# relative to the default repo.
function test_load_starlark_from_external_repo_with_repo_relative_label_load() {
  run_external_starlark_load_test "@//another_local_pkg:local_constants.bzl" \
    "LOCAL!"
}

function test_starlark_repository_relative_label() {
  repo2=$TEST_TMPDIR/repo2
  mkdir -p $repo2
  touch $repo2/WORKSPACE $repo2/BUILD
  cat > $repo2/remote.bzl <<EOF
def _impl(ctx):
  print(Label("//foo:bar"))

remote_rule = rule(
    implementation = _impl,
)
EOF

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(
    name = "r",
    path = "$repo2",
)
EOF
  cat > BUILD <<EOF
load('@r//:remote.bzl', 'remote_rule')

remote_rule(name = 'local')
EOF

  bazel build //:local &> $TEST_log || fail "Building local failed"
  expect_log "@r//foo:bar"
}

# Going one level deeper: if we have:
# local/
#   BUILD
# r1/
#   BUILD
# r2/
#   BUILD
#   remote.bzl
# If //foo in local depends on //bar in r1, which is a Starlark rule defined in
# r2/remote.bzl, then a Label in remote.bzl should resolve to @r2//whatever.
function test_starlark_repository_nested_relative_label() {
  repo1=$TEST_TMPDIR/repo1
  repo2=$TEST_TMPDIR/repo2
  mkdir -p $repo1 $repo2

  # local
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(
    name = "r1",
    path = "$repo1",
)
local_repository(
    name = "r2",
    path = "$repo2",
)
EOF
  cat > BUILD <<'EOF'
genrule(
    name = "foo",
    srcs = ["@r1//:bar"],
    outs = ["foo.out"],
    cmd = "echo '$(SRCS)' > $@",
)
EOF

  # r1
  touch $repo1/WORKSPACE
  cat > $repo1/BUILD <<EOF
load('@r2//:remote.bzl', 'remote_rule')

remote_rule(
    name = 'bar',
    visibility = ["//visibility:public"]
)
EOF

  # r2
  touch $repo2/WORKSPACE $repo2/BUILD
  cat > $repo2/remote.bzl <<EOF
def _impl(ctx):
  print(Label("//foo:bar"))

remote_rule = rule(
    implementation = _impl,
)
EOF

  bazel build //:foo &> $TEST_log || fail "Building local failed"
  expect_log "@r2//foo:bar"
}

function test_aspects_and_starlark_repositories() {
cat > WORKSPACE <<EOF
bind(name="x1", actual="//:x1")
load("//:repo.bzl", "repo")
bind(name="x2", actual="//:x2")
EOF

cat > BUILD <<EOF
load("//:rule.bzl", "test_rule")

filegroup(name = "x1", visibility = ["//visibility:public"])
filegroup(name = "x2", visibility = ["//visibility:public"])
test_rule(
    name = "tr",
    deps = ["//external:x1", "//external:x2"],
)
EOF

cat > repo.bzl <<EOF
def repo():
  pass
EOF

cat > rule.bzl <<EOF
def test_aspect_impl(target, ctx):
  return struct()

test_aspect = aspect(
    attrs = {
        "_x": attr.label_list(default = [
            Label("//external:x1"),
            Label("//external:x2"),
        ]),
    },
    implementation = test_aspect_impl,
)

def test_rule_impl(ctx):
  return struct()

test_rule = rule(
    attrs = {
        "deps": attr.label_list(
            allow_files = True,
            allow_rules = [
                "filegroup",
            ],
            aspects = [test_aspect],
        ),
    },
    implementation = test_rule_impl,
)
EOF

  bazel build //:tr || fail "build failed"
}

run_suite "Test Starlark loads from/in external repositories"
