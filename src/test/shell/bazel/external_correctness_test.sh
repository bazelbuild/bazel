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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  LOCAL=$(pwd)
  REMOTE=$TEST_TMPDIR/remote

  # Set up empty remote repo.
  mkdir -p $REMOTE
  touch $REMOTE/REPO.bazel
  cat > $REMOTE/BUILD <<EOF
genrule(
    name = "get-input",
    outs = ["an-input"],
    srcs = ["input"],
    cmd = "cat \$< > \$@",
    visibility = ["//visibility:public"],
)
EOF

  # Set up local repo that uses $REMOTE as an external repo.
  cat > $LOCAL/MODULE.bazel <<EOF
local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(
    name = "a",
    path = "$REMOTE",
)
EOF
  cat > $LOCAL/BUILD <<EOF
genrule(
    name = "b",
    srcs = ["@a//:get-input"],
    outs = ["b.out"],
    cmd = "cat \$< > \$@",
)
EOF
}

function test_build_file_changes_are_noticed() {
  cat > $REMOTE/BUILD <<EOF
SYNTAX ERROR
EOF
  bazel build //:b &> $TEST_log && fail "Build succeeded"
  expect_log "syntax error at 'ERROR'"

  cat > $REMOTE/BUILD <<EOF
genrule(
    name = "get-input",
    outs = ["a.out"],
    cmd = "echo 'I come from @a' > \$@",
    visibility = ["//visibility:public"],
)
EOF

  bazel build //:b &> $TEST_log || fail "Build failed"
  assert_contains "I come from @a" bazel-genfiles/b.out
}

function test_external_file_changes_are_noticed() {
  version="1.0"
  cat > $REMOTE/input <<EOF
$version
EOF
  bazel build //:b &> $TEST_log || fail "Build failed"
  assert_contains $version bazel-genfiles/b.out

  version="2.0"
  cat > $REMOTE/input <<EOF
$version
EOF
  bazel build //:b &> $TEST_log || fail "Build failed"
  assert_contains $version bazel-genfiles/b.out
}

function test_symlink_changes_are_noticed() {
  cat > $REMOTE/version1 <<EOF
1.0
EOF
  cat > $REMOTE/version2 <<EOF
2.0
EOF
  rm $REMOTE/input
  ln -s $REMOTE/version1 $REMOTE/input
  bazel build //:b &> $TEST_log || fail "Build failed"
  assert_contains 1.0 bazel-genfiles/b.out

  rm $REMOTE/input
  ln -s $REMOTE/version2 $REMOTE/input
  bazel build //:b &> $TEST_log || fail "Build failed"
  assert_contains 2.0 bazel-genfiles/b.out
}

function test_parent_symlink_change() {
  REMOTE1=$TEST_TMPDIR/remote1
  REMOTE2=$TEST_TMPDIR/remote2
  mkdir -p $REMOTE1 $REMOTE2
  cp -R $REMOTE/* $REMOTE1
  cp -R $REMOTE/* $REMOTE2
  cat > $REMOTE1/input <<EOF
1.0
EOF
  cat > $REMOTE2/input <<EOF
2.0
EOF
  rm -rf $REMOTE
  ln -s $REMOTE1 $REMOTE

  bazel build //:b &> $TEST_log || fail "Build failed"
  assert_contains 1.0 bazel-genfiles/b.out

  rm $REMOTE
  ln -s $REMOTE2 $REMOTE
  bazel build //:b &> $TEST_log || fail "Build failed"
  assert_contains 2.0 bazel-genfiles/b.out
}

function test_genrule_d_correctness() {
  subdir=$REMOTE/b/c
  mkdir -p $subdir
  cat > $subdir/BUILD <<EOF
genrule(
    name = "echo-d",
    outs = ["d"],
    cmd = "echo \$(@D) > \$@",
)
EOF
  bazel build @a//b/c:echo-d &> $TEST_log || fail "Build failed"
  assert_contains "bazel-out/.*-fastbuild/.*/external/_main~_repo_rules~a/b/c" \
    "bazel-genfiles/external/_main~_repo_rules~a/b/c/d"
}

function test_package_group_in_external_repos() {
  REMOTE=$TEST_TMPDIR/r
  mkdir -p $REMOTE/v $REMOTE/a v a
  touch $REMOTE/REPO.bazel

  echo 'filegroup(name="rv", srcs=["//:fg"])' > $REMOTE/v/BUILD
  echo 'filegroup(name="ra", srcs=["//:fg"])' > $REMOTE/a/BUILD
  echo 'filegroup(name="mv", srcs=["@r//:fg"])' > v/BUILD
  echo 'filegroup(name="ma", srcs=["@r//:fg"])' > a/BUILD
  cat > $REMOTE/BUILD <<EOF
package_group(name="pg", packages=["//v"])
filegroup(name="fg", visibility=[":pg"])
EOF

  echo "local_repository(name='r', path='$REMOTE')" >> MODULE.bazel
  bazel build @r//v:rv >& $TEST_log || fail "Build failed"
  bazel build @r//a:ra >& $TEST_log && fail "Build succeeded"
  expect_log "target '@@_main~_repo_rules~r//:fg' is not visible"
  bazel build //a:ma >& $TEST_log && fail "Build succeeded"
  expect_log "target '@@_main~_repo_rules~r//:fg' is not visible"
  bazel build //v:mv >& $TEST_log && fail "Build succeeded"
  expect_log "target '@@_main~_repo_rules~r//:fg' is not visible"

}

# Regression test for #517.
function test_refs_btwn_repos() {
  REMOTE1=$TEST_TMPDIR/remote1
  REMOTE2=$TEST_TMPDIR/remote2
  mkdir -p $REMOTE1 $REMOTE2
  touch $REMOTE1/REPO.bazel $REMOTE2/REPO.bazel
  cat > $REMOTE1/input <<EOF
1.0
EOF
  cat > $REMOTE1/BUILD <<EOF
exports_files(['input'])
EOF
  cat > $REMOTE2/BUILD <<EOF
genrule(
    name = "x",
    srcs = ["@remote1//:input"],
    cmd = "cat \$< > \$@",
    outs = ["x.out"],
)
EOF
  cat >> MODULE.bazel <<EOF
local_repository(
    name = "remote1",
    path = "$REMOTE1",
)
local_repository(
    name = "remote2",
    path = "$REMOTE2",
)
EOF

  bazel build @remote2//:x &> $TEST_log || fail "Build failed"
  assert_contains 1.0 bazel-genfiles/external/_main~_repo_rules~remote2/x.out
}

function test_visibility_attributes_in_external_repos() {
  REMOTE=$TEST_TMPDIR/r
  mkdir -p $REMOTE/v $REMOTE/r
  touch $REMOTE/REPO.bazel

  cat > $REMOTE/r/BUILD <<EOF
package(default_visibility=["//v:v"])
filegroup(name='fg1')  # Inherits default visibility
filegroup(name='fg2', visibility=["//v:v"])
EOF

  cat > $REMOTE/v/BUILD <<EOF
package_group(name="v", packages=["//"])
EOF

  cat >$REMOTE/BUILD <<EOF
filegroup(name="fg", srcs=["//r:fg1", "//r:fg2"])
EOF

  cat >> MODULE.bazel <<EOF
local_repository(name = "r", path = "$REMOTE")
EOF

  cat > BUILD <<EOF
filegroup(name="fg", srcs=["@r//r:fg1", "@r//r:fg2"])
EOF

  bazel build @r//:fg || fail "Build failed"
  bazel build //:fg >& $TEST_log && fail "Build succeeded"
  expect_log "target '@@_main~_repo_rules~r//r:fg1' is not visible"

}

function test_select_in_external_repo() {
  REMOTE=$TEST_TMPDIR/r
  mkdir -p $REMOTE/a $REMOTE/c d
  touch $REMOTE/REPO.bazel

  cat > $REMOTE/a/BUILD <<'EOF'
genrule(
    name = "gr",
    srcs = [],
    outs = ["gro"],
    cmd = select({
      "//c:one": "echo one > $@",
      ":two": "echo two > $@",
      "@//d:three": "echo three > $@",
      "@//:four": "echo four > $@",
      "//conditions:default": "echo default > $@",
    }))

config_setting(name = "two", values = { "define": "ARG=two" })
EOF

  cat > $REMOTE/c/BUILD <<EOF
package(default_visibility=["//visibility:public"])
config_setting(name = "one", values = { "define": "ARG=one" })
EOF

  cat >> MODULE.bazel <<EOF
local_repository(name="r", path="$REMOTE")
EOF

  cat > d/BUILD <<EOF
package(default_visibility=["//visibility:public"])
config_setting(name = "three", values = { "define": "ARG=three" })
EOF

  cat > BUILD <<EOF
package(default_visibility=["//visibility:public"])
config_setting(name = "four", values = { "define": "ARG=four" })
EOF

  bazel build @r//a:gr || fail "build failed"
  assert_contains "default" bazel-genfiles/external/_main~_repo_rules~r/a/gro
  bazel build @r//a:gr --define=ARG=one|| fail "build failed"
  assert_contains "one" bazel-genfiles/external/_main~_repo_rules~r/a/gro
  bazel build @r//a:gr --define=ARG=two || fail "build failed"
  assert_contains "two" bazel-genfiles/external/_main~_repo_rules~r/a/gro
  bazel build @r//a:gr --define=ARG=three || fail "build failed"
  assert_contains "three" bazel-genfiles/external/_main~_repo_rules~r/a/gro
  bazel build @r//a:gr --define=ARG=four || fail "build failed"
  assert_contains "four" bazel-genfiles/external/_main~_repo_rules~r/a/gro

}

function top_level_dir_changes_helper() {
  batch_flag="$1"
  mkdir -p r/subdir m
  touch r/one r/subdir/two

  cat > m/MODULE.bazel <<'EOF'
new_local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")
new_local_repository(
    name = "r",
    path = "../r",
    build_file_content = """
genrule(
    name = "fg",
    cmd = "ls $(SRCS) > $@",
    srcs=glob(["**"]),
    outs = ["fg.out"],
)""",
)
EOF
  write_default_lockfile "m/MODULE.bazel.lock"
  cd m
  bazel "$batch_flag" build @r//:fg &> $TEST_log || \
    fail "Expected build to succeed"
  touch ../r/three
  bazel "$batch_flag" build @r//:fg &> $TEST_log || \
    fail "Expected build to succeed"
  assert_contains "external/_main~_repo_rules~r/three" bazel-genfiles/external/_main~_repo_rules~r/fg.out
  touch ../r/subdir/four
  bazel "$batch_flag" build @r//:fg &> $TEST_log || \
    fail "Expected build to succeed"
  assert_contains "external/_main~_repo_rules~r/subdir/four" bazel-genfiles/external/_main~_repo_rules~r/fg.out
}

function test_top_level_dir_changes_batch() {
  top_level_dir_changes_helper --batch
}

function test_top_level_dir_changes_nobatch() {
  top_level_dir_changes_helper --nobatch
}

function test_non_extsietnt_repo_in_pattern() {
  bazel build @non_existent_repo//... &> $TEST_log && fail "Expected build to fail"
  expect_log "ERROR: No repository visible as '@non_existent_repo' from main repository"
}

run_suite "//external correctness tests"
