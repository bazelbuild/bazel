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
# Test git_repository and new_git_repository workspace rules.
#

set -euo pipefail
# --- begin runfiles.bash initialization ---
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  if [[ -f "$0.runfiles_manifest" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
  elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
    export RUNFILES_DIR="$0.runfiles"
  fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  # As of 2019-01-15, Bazel on Windows only supports MSYS Bash.
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if $is_windows; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
  # Enable symlink runfiles tree to make bazel run work
  add_to_bazelrc "build --enable_runfiles"
fi

# Global test setup.
#
# Unpacks the test Git repositories in the test temporary directory.
function set_up() {
  bazel clean --expunge
  local repos_dir=$TEST_TMPDIR/repos
  if [ -e "$repos_dir" ]; then
    rm -rf $repos_dir
  fi

  mkdir -p $repos_dir
  cp "$(rlocation io_bazel/src/test/shell/bazel/testdata/pluto-repo.tar.gz)" $repos_dir
  cp "$(rlocation io_bazel/src/test/shell/bazel/testdata/outer-planets-repo.tar.gz)" $repos_dir
  cp "$(rlocation io_bazel/src/test/shell/bazel/testdata/refetch-repo.tar.gz)" $repos_dir
  cd $repos_dir
  tar zxf pluto-repo.tar.gz
  tar zxf outer-planets-repo.tar.gz
  tar zxf refetch-repo.tar.gz

  local cache_dir=/tmp/bazel_cache
  rm -rf $cache_dir

  # Fix environment variables for a hermetic use of git.
  export GIT_CONFIG_NOSYSTEM=1
  export GIT_CONFIG_NOGLOBAL=1
  export HOME=
  export XDG_CONFIG_HOME=
}

# Test cloning a Git repository using the git_repository rule.
#
# This test uses the pluto Git repository at tag 1-build, which contains the
# following files:
#
# pluto/
#   WORKSPACE
#   BUILD
#   info
#
# Followed by a test at 2-subdir which contains the following files to test
# the strip_prefix functionality:
#
# pluto/
#   pluto/
#     WORKSPACE
#     BUILD
#     info
#
# In each case, set up workspace with the following files:
#
# $WORKSPACE_DIR/
#   WORKSPACE
#   planets/
#     BUILD
#     planet_info.sh
#
# //planets has a dependency on a target in the pluto Git repository.
function do_git_repository_test() {
  local pluto_repo_dir=$TEST_TMPDIR/repos/pluto
  # Commit corresponds to tag 1-build. See testdata/pluto.git_log.
  local commit_hash="$1"
  local strip_prefix=""
  local shallow_since=""
  [ $# -eq 2 ] && strip_prefix="strip_prefix=\"$2\","
  [ $# -eq 3 ] && shallow_since="shallow_since=\"$3\","
  # Create a workspace that clones the repository at the first commit.
  cd $WORKSPACE_DIR
  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(
    name = "pluto",
    remote = "$pluto_repo_dir",
    commit = "$commit_hash",
    $strip_prefix
    $shallow_since
)
EOF
  mkdir -p planets
  cat > planets/BUILD <<EOF
sh_binary(
    name = "planet-info",
    srcs = ["planet_info.sh"],
    data = ["@pluto//:pluto"],
)
EOF

  cat > planets/planet_info.sh <<EOF
#!/bin/sh
cat ../pluto/info
EOF
  chmod +x planets/planet_info.sh

  bazel run //planets:planet-info >& $TEST_log \
    || echo "Expected build/run to succeed"
  expect_log "Pluto is a dwarf planet"
}

# This test:
# 1. Creates a git_repository rule with some commit hash
# 2. Make sure worksapce can be built and repository cache is updated
function test_git_repositry_cache_is_populated() {
  local pluto_repo_dir=$TEST_TMPDIR/repos/pluto
  local cache_dir=/tmp/bazel_cache
  local commit_hash="52f9a3f87a2dd17ae0e5847bbae9734f09354afd"


  export BAZEL_GIT_REPOSITORY_CACHE=$cache_dir

  do_git_repository_test $commit_hash

  cache_commit_hash="$(cat $cache_dir/worktrees/pluto/HEAD)"
  assert_equals $commit_hash $cache_commit_hash
}

function test_git_repository() {
  do_git_repository_test "52f9a3f87a2dd17ae0e5847bbae9734f09354afd"
}

function test_git_repository_strip_prefix() {
  # This commit has the files in a subdirectory named 'pluto'
  # so we strip_prefix and the build should still work.
  do_git_repository_test "dbf9236251a9ea01b7a2eb563ca8e911060fc97c" "pluto"
}

function test_git_repository_shallow_since() {
    # This date is the day the commit was made.
    do_git_repository_test "52f9a3f87a2dd17ae0e5847bbae9734f09354afd" "" "2016-07-16"
}
function test_new_git_repository_with_build_file() {
  do_new_git_repository_test "0-initial" "build_file"
}

function test_new_git_repository_with_build_file_strip_prefix() {
  do_new_git_repository_test "3-subdir-bare" "build_file" "pluto"
}

function test_new_git_repository_with_build_file_content() {
  do_new_git_repository_test "0-initial" "build_file_content"
}

function test_new_git_repository_with_build_file_content_strip_prefix() {
  do_new_git_repository_test "3-subdir-bare" "build_file_content" "pluto"
}

# Test cloning a Git repository using the new_git_repository rule.
#
# This test uses the pluto Git repository at tag 0-initial, which contains the
# following files:
#
# pluto/
#   info
#
# Then it uses the pluto Git repository at tag 3-subdir-bare, which contains the
# following files:
# pluto/
#   pluto/
#     info
#
# Set up workspace with the following files:
#
# $WORKSPACE_DIR/
#   WORKSPACE
#   pluto.BUILD
#   planets/
#     BUILD
#     planet_info.sh
#
# //planets has a dependency on a target in the $TEST_TMPDIR/pluto Git
# repository.
function do_new_git_repository_test() {
  local pluto_repo_dir=$TEST_TMPDIR/repos/pluto
  local strip_prefix=""
  [ $# -eq 3 ] && strip_prefix="strip_prefix=\"$3\","

  # Create a workspace that clones the repository at the first commit.
  cd $WORKSPACE_DIR

  if [ "$2" == "build_file" ] ; then
    cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'new_git_repository')
new_git_repository(
    name = "pluto",
    remote = "$pluto_repo_dir",
    tag = "$1",
    build_file = "//:pluto.BUILD",
    $strip_prefix
)
EOF

  cat > BUILD <<EOF
exports_files(['pluto.BUILD'])
EOF
    cat > pluto.BUILD <<EOF
filegroup(
    name = "pluto",
    srcs = ["info"],
    visibility = ["//visibility:public"],
)
EOF
  else
    cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'new_git_repository')
new_git_repository(
    name = "pluto",
    remote = "$pluto_repo_dir",
    tag = "$1",
    $strip_prefix
    build_file_content = """
filegroup(
    name = "pluto",
    srcs = ["info"],
    visibility = ["//visibility:public"],
)"""
)
EOF
  fi

  mkdir -p planets
  cat > planets/BUILD <<EOF
sh_binary(
    name = "planet-info",
    srcs = ["planet_info.sh"],
    data = ["@pluto//:pluto"],
)
EOF

  cat > planets/planet_info.sh <<EOF
#!/bin/sh
cat ../pluto/info
EOF
  chmod +x planets/planet_info.sh

  bazel run //planets:planet-info >& $TEST_log \
      || echo "Expected build/run to succeed"
  if [ "$1" == "0-initial" ]; then
      expect_log "Pluto is a planet"
  else
      expect_log "Pluto is a dwarf planet"
  fi
}

# Test cloning a Git repository that has a submodule using the
# new_git_repository rule.
#
# This test uses the outer-planets Git repository at revision 1-submodule, which
# contains the following files:
#
# outer_planets/
#   neptune/
#     info
#   pluto/  --> submodule ../pluto
#     info
#
# Set up workspace with the following files:
#
# $WORKSPACE_DIR/
#   WORKSPACE
#   outer_planets.BUILD
#   planets/
#     BUILD
#     planet_info.sh
#
# planets has a dependency on targets in the $TEST_TMPDIR/outer_planets Git
# repository.
function test_new_git_repository_submodules() {
  local outer_planets_repo_dir=$TEST_TMPDIR/repos/outer-planets

  # Create a workspace that clones the outer_planets repository.
  cd $WORKSPACE_DIR
  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'new_git_repository')
new_git_repository(
    name = "outer_planets",
    remote = "$outer_planets_repo_dir",
    tag = "1-submodule",
    init_submodules = 1,
    build_file = "//:outer_planets.BUILD",
)
EOF

  cat > BUILD <<EOF
exports_files(['outer_planets.BUILD'])
EOF
  cat > outer_planets.BUILD <<EOF
filegroup(
    name = "neptune",
    srcs = ["neptune/info"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "pluto",
    srcs = ["pluto/info"],
    visibility = ["//visibility:public"],
)
EOF

  mkdir -p planets
  cat > planets/BUILD <<EOF
sh_binary(
    name = "planet-info",
    srcs = ["planet_info.sh"],
    data = [
        "@outer_planets//:neptune",
        "@outer_planets//:pluto",
    ],
)
EOF

  cat > planets/planet_info.sh <<EOF
#!/bin/sh
cat ../outer_planets/neptune/info
cat ../outer_planets/pluto/info
EOF
  chmod +x planets/planet_info.sh

  bazel run //planets:planet-info >& $TEST_log \
    || echo "Expected build/run to succeed"
  expect_log "Neptune is a planet"
  expect_log "Pluto is a planet"
}

function test_git_repository_not_refetched_on_server_restart() {
  local repo_dir=$TEST_TMPDIR/repos/refetch

  cd $WORKSPACE_DIR
  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='22095302abaf776886879efa5129aa4d44c53017', verbose=True)
EOF

  # Use batch to force server restarts.
  bazel --batch build @g//:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 1" bazel-genfiles/external/g/go

  # Without changing anything, restart the server, which should not cause the checkout to be re-cloned.
  bazel --batch build @g//:g >& $TEST_log || fail "Build failed"
  expect_not_log "Cloning"
  assert_contains "GIT 1" bazel-genfiles/external/g/go

  # Change the commit id, which should cause the checkout to be re-cloned.
  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='db134ae9b644d8237954a8e6f1ef80fcfd85d521', verbose=True)
EOF

  bazel --batch build @g//:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 2" bazel-genfiles/external/g/go

  # Change the WORKSPACE but not the commit id, which should not cause the checkout to be re-cloned.
  cat > WORKSPACE <<EOF
# This comment line is to change the line numbers, which should not cause Bazel
# to refetch the repository
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='db134ae9b644d8237954a8e6f1ef80fcfd85d521', verbose=True)
EOF

  bazel --batch build @g//:g >& $TEST_log || fail "Build failed"
  expect_not_log "Cloning"
  assert_contains "GIT 2" bazel-genfiles/external/g/go
}

function test_git_repository_not_refetched_on_server_restart_strip_prefix() {
  local repo_dir=$TEST_TMPDIR/repos/refetch
  # Change the strip_prefix which should cause a new checkout
  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='17ea13b242e4cbcc27a6ef745939ebb7dcccea10', verbose=True)
EOF
  bazel --batch build @g//gdir:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 2" bazel-genfiles/external/g/gdir/go

  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='17ea13b242e4cbcc27a6ef745939ebb7dcccea10', verbose=True, strip_prefix="gdir")
EOF
  bazel --batch build @g//:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 2" bazel-genfiles/external/g/go
}


function test_git_repository_refetched_when_commit_changes() {
  local repo_dir=$TEST_TMPDIR/repos/refetch

  cd $WORKSPACE_DIR
  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='22095302abaf776886879efa5129aa4d44c53017', verbose=True)
EOF

  bazel build @g//:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 1" bazel-genfiles/external/g/go

  # Change the commit id, which should cause the checkout to be re-cloned.
  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='db134ae9b644d8237954a8e6f1ef80fcfd85d521', verbose=True)
EOF

  bazel build @g//:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 2" bazel-genfiles/external/g/go
}

function test_git_repository_and_nofetch() {
  local repo_dir=$TEST_TMPDIR/repos/refetch

  cd $WORKSPACE_DIR
  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='22095302abaf776886879efa5129aa4d44c53017')
EOF

  bazel build --nofetch @g//:g >& $TEST_log && fail "Build succeeded"
  expect_log "fetching repositories is disabled"
  bazel build @g//:g >& $TEST_log || fail "Build failed"
  assert_contains "GIT 1" bazel-genfiles/external/g/go

  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='db134ae9b644d8237954a8e6f1ef80fcfd85d521')
EOF


  bazel build --nofetch @g//:g >& $TEST_log || fail "Build failed"
  expect_log "External repository 'g' is not up-to-date"
  assert_contains "GIT 1" bazel-genfiles/external/g/go
  bazel build  @g//:g >& $TEST_log || fail "Build failed"
  assert_contains "GIT 2" bazel-genfiles/external/g/go

  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='17ea13b242e4cbcc27a6ef745939ebb7dcccea10', strip_prefix="gdir")
EOF

  bazel build --nofetch @g//:g >& $TEST_log || fail "Build failed"
  expect_log "External repository 'g' is not up-to-date"
  bazel build  @g//:g >& $TEST_log || fail "Build failed"
  assert_contains "GIT 2" bazel-genfiles/external/g/go

}

# Helper function for setting up the workspace as follows
#
# $WORKSPACE_DIR/
#   WORKSPACE
#   planets/
#     planet_info.sh
#     BUILD
function setup_error_test() {
  cd $WORKSPACE_DIR
  mkdir -p planets
  cat > planets/planet_info.sh <<EOF
#!/bin/sh
cat external/pluto/info
EOF

  cat > planets/BUILD <<EOF
sh_binary(
    name = "planet-info",
    srcs = ["planet_info.sh"],
    data = ["@pluto//:pluto"],
)
EOF
}

# Verifies that rule fails if both tag and commit are set.
#
# This test uses the pluto Git repository at tag 1-build, which contains the
# following files:
#
# pluto/
#   WORKSPACE
#   BUILD
#   info
function test_git_repository_both_commit_tag_error() {
  setup_error_test
  local pluto_repo_dir=$TEST_TMPDIR/repos/pluto
  # Commit corresponds to tag 1-build. See testdata/pluto.git_log.
  local commit_hash="52f9a3f87a2dd17ae0e5847bbae9734f09354afd"

  cd $WORKSPACE_DIR
  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(
    name = "pluto",
    remote = "$pluto_repo_dir",
    tag = "1-build",
    commit = "$commit_hash",
)
EOF

  bazel fetch //planets:planet-info >& $TEST_log \
    || echo "Expect run to fail."
  expect_log "Exactly one of commit"
}

# Verifies that rule fails if neither tag or commit are set.
#
# This test uses the pluto Git repository at tag 1-build, which contains the
# following files:
#
# pluto/
#   WORKSPACE
#   BUILD
#   info
function test_git_repository_no_commit_tag_error() {
  setup_error_test
  local pluto_repo_dir=$TEST_TMPDIR/repos/pluto

  cd $WORKSPACE_DIR
  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(
    name = "pluto",
    remote = "$pluto_repo_dir",
)
EOF

  bazel fetch //planets:planet-info >& $TEST_log \
    || echo "Expect run to fail."
  expect_log "Exactly one of commit"
}

# Verifies that if a non-existent subdirectory is supplied, then strip_prefix
# throws an error.
function test_invalid_strip_prefix_error() {
  setup_error_test
  local pluto_repo_dir=$TEST_TMPDIR/repos/pluto

  cd $WORKSPACE_DIR
  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(
    name = "pluto",
    remote = "$pluto_repo_dir",
    tag = "1-build",
    strip_prefix = "dir_does_not_exist"
)
EOF

  bazel fetch //planets:planet-info >& $TEST_log \
    || echo "Expect run to fail."
  expect_log "strip_prefix at dir_does_not_exist does not exist in repo"
}


# Verifies that rule fails if tag and shallow_since are set
#
function test_git_repository_shallow_since_with_tag_error() {
  setup_error_test
  local pluto_repo_dir=$TEST_TMPDIR/pluto

  cd $WORKSPACE_DIR
  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(
    name = "pluto",
    remote = "$pluto_repo_dir",
    tag = "1-build",
    shallow_since = "2018-01-01"
)
EOF

  bazel fetch //planets:planet-info >& $TEST_log \
    || echo "Expect run to fail."
  expect_log "shallow_since not allowed if a tag is specified; --depth=1 will be used for tags"
}

# Verifies that rule fails if you target a commit that is before
# the shallow point
#
function test_git_repository_shallow_since_with_earlier_commit_error() {
  setup_error_test
  local pluto_repo_dir=$TEST_TMPDIR/pluto

  cd $WORKSPACE_DIR
  # This commit was made in July so should not be available if we
  # shallow since December.
  cat > WORKSPACE <<EOF
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(
    name = "pluto",
    remote = "$pluto_repo_dir",
    commit = "52f9a3f87a2dd17ae0e5847bbae9734f09354afd",
    shallow_since = "2017-12-27"
)
EOF

  bazel fetch //planets:planet-info >& $TEST_log \
    || echo "Expect run to fail."
  expect_log "Error cloning"
}

run_suite "skylark git_repository tests"
