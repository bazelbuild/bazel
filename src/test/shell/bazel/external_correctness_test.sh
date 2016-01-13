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

source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

function set_up() {
  LOCAL=$(pwd)
  REMOTE=$TEST_TMPDIR/remote

  # Set up empty remote repo.
  mkdir -p $REMOTE
  touch $REMOTE/WORKSPACE
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
  cat > $LOCAL/WORKSPACE <<EOF
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
  assert_contains "bazel-out/local_.*-fastbuild/genfiles/external/a/b/c" \
    "bazel-genfiles/external/a/b/c/d"
}

# Regression test for #517.
function test_refs_btwn_repos() {
  REMOTE1=$TEST_TMPDIR/remote1
  REMOTE2=$TEST_TMPDIR/remote2
  mkdir -p $REMOTE1 $REMOTE2
  touch $REMOTE1/WORKSPACE $REMOTE2/WORKSPACE
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
  cat > WORKSPACE <<EOF
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
  assert_contains 1.0 bazel-genfiles/external/remote2/x.out
}

function test_visibility_in_external_repo() {
  REMOTE=$TEST_TMPDIR/r
  mkdir -p $REMOTE/v

  cat > $REMOTE/BUILD <<EOF
package(default_visibility=["//v:v"])
filegroup(name='fg1')  # Inherits default visibility
filegroup(name='fg2', visibility=["//v:v"])
EOF

  cat > $REMOTE/v/BUILD <<EOF
package_group(name="v", packages=["//"])
EOF

  cat > WORKSPACE <<EOF
local_repository(name = "r", path = "$REMOTE")
EOF

  cat > BUILD <<EOF
filegroup(name = "fg", srcs=["@r//:fg1", "@r//:fg2"])
EOF

  bazel build //:fg || fail "Build failed"
}

run_suite "//external correctness tests"
