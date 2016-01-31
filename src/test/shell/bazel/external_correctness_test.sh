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

function test_package_group_in_external_repos() {
  REMOTE=$TEST_TMPDIR/r
  mkdir -p $REMOTE/v $REMOTE/a v a

  echo 'filegroup(name="rv", srcs=["//:fg"])' > $REMOTE/v/BUILD
  echo 'filegroup(name="ra", srcs=["//:fg"])' > $REMOTE/a/BUILD
  echo 'filegroup(name="mv", srcs=["@r//:fg"])' > v/BUILD
  echo 'filegroup(name="ma", srcs=["@r//:fg"])' > a/BUILD
  cat > $REMOTE/BUILD <<EOF
package_group(name="pg", packages=["//v"])
filegroup(name="fg", visibility=[":pg"])
EOF

  echo "local_repository(name='r', path='$REMOTE')" > WORKSPACE
  bazel build @r//v:rv >& $TEST_log || fail "Build failed"
  bazel build @r//a:ra >& $TEST_log && fail "Build succeeded"
  expect_log "Target '@r//:fg' is not visible"
  bazel build //a:ma >& $TEST_log && fail "Build succeeded"
  expect_log "Target '@r//:fg' is not visible"
  bazel build //v:mv >& $TEST_log && fail "Build succeeded"
  expect_log "Target '@r//:fg' is not visible"

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

function test_visibility_attributes_in_external_repos() {
  REMOTE=$TEST_TMPDIR/r
  mkdir -p $REMOTE/v $REMOTE/r

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

  cat > WORKSPACE <<EOF
local_repository(name = "r", path = "$REMOTE")
EOF

  cat > BUILD <<EOF
filegroup(name="fg", srcs=["@r//r:fg1", "@r//r:fg2"])
EOF

  bazel build @r//:fg || fail "Build failed"
  bazel build //:fg >& $TEST_log && fail "Build succeeded"
  expect_log "Target '@r//r:fg1' is not visible"

}

function test_select_in_external_repo() {
  REMOTE=$TEST_TMPDIR/r
  mkdir -p $REMOTE/a $REMOTE/c d

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

  cat > WORKSPACE <<EOF
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
  assert_contains "default" bazel-genfiles/external/r/a/gro
  bazel build @r//a:gr --define=ARG=one|| fail "build failed"
  assert_contains "one" bazel-genfiles/external/r/a/gro
  bazel build @r//a:gr --define=ARG=two || fail "build failed"
  assert_contains "two" bazel-genfiles/external/r/a/gro
  bazel build @r//a:gr --define=ARG=three || fail "build failed"
  assert_contains "three" bazel-genfiles/external/r/a/gro
  bazel build @r//a:gr --define=ARG=four || fail "build failed"
  assert_contains "four" bazel-genfiles/external/r/a/gro

}

run_suite "//external correctness tests"
