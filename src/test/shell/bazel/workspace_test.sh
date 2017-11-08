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

export JAVA_RUNFILES=$BAZEL_RUNFILES

function setup_repo() {
  mkdir -p $1
  touch $1/WORKSPACE
  echo $2 > $1/thing
  cat > $1/BUILD <<EOF
genrule(
    name = "x",
    srcs = ["thing"],
    cmd = "cat \$(location thing) > \$@",
    outs = ["out"],
)
EOF
}

function test_workspace_changes() {
  repo_a=$TEST_TMPDIR/a
  repo_b=$TEST_TMPDIR/b
  setup_repo $repo_a hi
  setup_repo $repo_b bye

  cat > WORKSPACE <<EOF
local_repository(
    name = "x",
    path = "$repo_a",
)
EOF

  bazel build @x//:x || fail "build failed"
  assert_contains "hi" bazel-genfiles/external/x/out

  cat > WORKSPACE <<EOF
local_repository(
    name = "x",
    path = "$repo_b",
)
EOF

  bazel build @x//:x || fail "build failed"
  assert_contains "bye" bazel-genfiles/external/x/out
}


function test_path_with_spaces() {
  ws="a b"
  mkdir "$ws"
  cd "$ws"
  touch WORKSPACE

  bazel info &> $TEST_log && fail "Info succeeeded"
  bazel help &> $TEST_log || fail "Help failed"
}

# Tests for middleman conflict when using workspace repository
function test_middleman_conflict() {
  local test_repo1=$TEST_TMPDIR/repo1
  local test_repo2=$TEST_TMPDIR/repo2

  mkdir -p $test_repo1
  mkdir -p $test_repo2
  echo "1" >$test_repo1/test.in
  echo "2" >$test_repo2/test.in
  echo 'filegroup(name="test", srcs=["test.in"], visibility=["//visibility:public"])' \
    >$test_repo1/BUILD
  echo 'filegroup(name="test", srcs=["test.in"], visibility=["//visibility:public"])' \
    >$test_repo2/BUILD
  touch $test_repo1/WORKSPACE
  touch $test_repo2/WORKSPACE

  cat > WORKSPACE <<EOF
local_repository(name = 'repo1', path='$test_repo1')
local_repository(name = 'repo2', path='$test_repo2')
EOF

  cat > BUILD <<'EOF'
genrule(
  name = "test",
  srcs = ["@repo1//:test", "@repo2//:test"],
  outs = ["test.out"],
  cmd = "cat $(SRCS) >$@"
)
EOF
  bazel fetch //:test || fail "Fetch failed"
  bazel build //:test || echo "Expected build to succeed"
  check_eq "12" "$(cat bazel-genfiles/test.out | tr -d '[:space:]')"
}

# Regression test for issue #724: NullPointerException in WorkspaceFile
function test_error_in_workspace_file() {
  # Create a buggy workspace
  cat >WORKSPACE <<'EOF'
/
EOF

  # Try to refer to the workspace.
  bazel --batch build @r//:rfg &>$TEST_log \
      && fail "Failure expected" || true

  expect_not_log "Exception"
}

function test_no_select() {
  cat > WORKSPACE <<EOF
new_local_repository(
    name = "foo",
    path = "/path/to/foo",
    build_file = select({
        "//x:y" : "BUILD.1",
        "//conditions:default" : "BUILD.2"}),
)
EOF

  bazel build @foo//... &> $TEST_log && fail "Failure expected" || true
  expect_log "select() cannot be used in WORKSPACE files"
}

function test_macro_select() {
  cat > WORKSPACE <<EOF
load('//:foo.bzl', 'foo_repo')
foo_repo()
EOF

  touch BUILD
  cat > foo.bzl <<EOF
def foo_repo():
  native.new_local_repository(
      name = "foo",
      path = "/path/to/foo",
      build_file = select({
          "//x:y" : "BUILD.1",
          "//conditions:default" : "BUILD.2"}),
  )
EOF

  bazel build @foo//... &> $TEST_log && fail "Failure expected" || true
  expect_log "select() cannot be used in macros called from WORKSPACE files"
}

function test_clean() {
  mkdir x
  cd x
  cat > WORKSPACE <<EOF
workspace(name = "y")
EOF
  cat > BUILD <<'EOF'
genrule(name = "z", cmd = "echo hi > $@", outs = ["x.out"], srcs = [])
EOF
  bazel build //:z &> $TEST_log || fail "Expected build to succeed"
  [ -L bazel-x ] || fail "bazel-x should be a symlink"
  bazel clean
  [ ! -L bazel-x ] || fail "bazel-x should have been removed"
}

function test_skylark_flags_affect_workspace() {
  cat > WORKSPACE <<EOF
load("//:macro.bzl", "macro")
print("In workspace: ")
macro()
EOF
  cat > macro.bzl <<EOF
def macro():
  print("In workspace macro: ")
EOF
  cat > BUILD <<'EOF'
genrule(name = "x", cmd = "echo hi > $@", outs = ["x.out"], srcs = [])
EOF

  MARKER="<== skylark flag test ==>"

  # Sanity check.
  bazel build //:x &>"$TEST_log" \
    || fail "Expected build to succeed"
  expect_log "In workspace: " "Did not find workspace print output"
  expect_log "In workspace macro: " "Did not find workspace macro print output"
  expect_not_log "$MARKER" \
    "Marker string '$MARKER' was seen even though \
    --internal_skylark_flag_test_canary wasn't passed"

  # Build with the special testing flag that appends a marker string to all
  # print() calls.
  bazel build //:x --internal_skylark_flag_test_canary &>"$TEST_log" \
    || fail "Expected build to succeed"
  expect_log "In workspace: $MARKER" \
    "Skylark flags are not propagating to workspace evaluation"
  expect_log "In workspace macro: $MARKER" \
    "Skylark flags are not propagating to workspace macro evaluation"
}

function test_workspace_name() {
  mkdir -p foo
  mkdir -p bar
  cat > foo/WORKSPACE <<EOF
workspace(name = "foo")

local_repository(
    name = "bar",
    path = "$PWD/bar",
)
EOF
  cat > foo/BUILD <<EOF
exports_files(glob(["*"]))
EOF
  touch foo/baz
  cat > bar/WORKSPACE <<EOF
workspace(name = "bar")
EOF
  cat > bar/BUILD <<EOF
genrule(
    name = "depend-on-foo",
    srcs = ["@foo//:baz"],
    cmd = "cat \$(SRCS) > \$@",
    outs = ["baz.out"],
)
EOF
  cd foo
  bazel build @bar//:depend-on-foo || fail "Expected build to succeed"
}

function test_workspace_override() {
  mkdir -p original
  touch original/WORKSPACE
  cat > original/BUILD <<'EOF'
genrule(
    name = "gen",
    cmd = "echo 'original' > $@",
    outs = ["gen.out"],
)
EOF

  mkdir -p override
  touch override/WORKSPACE
  cat > override/BUILD <<'EOF'
genrule(
    name = "gen",
    cmd = "echo 'override' > $@",
    outs = ["gen.out"],
)
EOF

  cat > WORKSPACE <<EOF
local_repository(
    name = "o",
    path = "original",
)
EOF
  bazel build --override_repository="o=$PWD/override" @o//:gen &> $TEST_log \
    || fail "Expected build to succeed"
  assert_contains "override" bazel-genfiles/external/o/gen.out

  bazel build @o//:gen &> $TEST_log \
    || fail "Expected build to succeed"
  assert_contains "original" bazel-genfiles/external/o/gen.out

  bazel build --override_repository="o=$PWD/override" @o//:gen &> $TEST_log \
    || fail "Expected build to succeed"
  assert_contains "override" bazel-genfiles/external/o/gen.out
}

run_suite "workspace tests"
