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
  # Enable symlink runfiles tree to make bazel run work
  add_to_bazelrc "build --enable_runfiles"
fi

# Global test setup.
#
# Unpacks the test Git repositories in the test temporary directory.
function set_up() {
  local repos_dir=$TEST_TMPDIR/repos
  if [ -e "$repos_dir" ]; then
    rm -rf $repos_dir
  fi

  mkdir -p $repos_dir
  cp "$(rlocation io_bazel/src/test/shell/bazel/testdata/pluto-repo.tar.gz)" $repos_dir
  cp "$(rlocation io_bazel/src/test/shell/bazel/testdata/outer-planets-repo.tar.gz)" $repos_dir
  cp "$(rlocation io_bazel/src/test/shell/bazel/testdata/refetch-repo.tar.gz)" $repos_dir
  cp "$(rlocation io_bazel/src/test/shell/bazel/testdata/strip-prefix-repo.tar.gz)" $repos_dir
  cd $repos_dir
  tar zxf pluto-repo.tar.gz
  tar zxf outer-planets-repo.tar.gz
  tar zxf refetch-repo.tar.gz
  tar zxf strip-prefix-repo.tar.gz

  setup_module_dot_bazel

  # Fix environment variables for a hermetic use of git.
  export GIT_CONFIG_NOSYSTEM=1
  export GIT_CONFIG_NOGLOBAL=1
  export HOME=
  export XDG_CONFIG_HOME=
}

# Shutdown Bazel so that we can safely delete files on Windows
function tear_down() {
  bazel shutdown
}

function get_pluto_repo() {
  echo "$TEST_TMPDIR/repos/pluto"
}

# Test cloning a Git repository using the git_repository rule.
#
# This test uses the pluto Git repository at tag 1-build, which contains the
# following files:
#
# pluto/
#   MODULE.bazel
#   BUILD
#   info
#
# Followed by a test at 2-subdir which contains the following files to test
# the strip_prefix functionality:
#
# pluto/
#   pluto/
#     MODULE.bazel
#     BUILD
#     info
#
# In each case, set up workspace with the following files:
#
# $WORKSPACE_DIR/
#   MODULE.bazel
#   planets/
#     BUILD
#     planet_info.sh
#
# //planets has a dependency on a target in the pluto Git repository.
function do_git_repository_test() {
  local pluto_repo_dir=$(get_pluto_repo)
  # Commit corresponds to tag 1-build. See testdata/pluto.git_log.
  local commit_hash="$1"
  local strip_prefix=""
  local shallow_since=""
  [ $# -eq 2 ] && strip_prefix="strip_prefix=\"$2\","
  [ $# -eq 3 ] && shallow_since="shallow_since=\"$3\","
  # Create a workspace that clones the repository at the first commit.
  cat >> MODULE.bazel <<EOF
git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
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
cat ../+_repo_rules+pluto/info
EOF
  chmod +x planets/planet_info.sh

  bazel run //planets:planet-info >& $TEST_log \
    || echo "Expected build/run to succeed"
  expect_log "Pluto is a dwarf planet"

  git_repos_count=$(find $(bazel info output_base)/external/+_repo_rules+pluto -type d -name .git | wc -l)
  assert_equals $git_repos_count 0
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
    # This date is the previous day before the commit was made.
    # We need the revious day, because git adds current time to the specified date.
    do_git_repository_test "52f9a3f87a2dd17ae0e5847bbae9734f09354afd" "" "2015-07-15"
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
#   MODULE.bazel
#   pluto.BUILD
#   planets/
#     BUILD
#     planet_info.sh
#
# //planets has a dependency on a target in the $TEST_TMPDIR/pluto Git
# repository.
function do_new_git_repository_test() {
  local pluto_repo_dir=$(get_pluto_repo)
  local strip_prefix=""
  [ $# -eq 3 ] && strip_prefix="strip_prefix=\"$3\","

  # Create a workspace that clones the repository at the first commit.

  if [ "$2" == "build_file" ] ; then
    cat >> MODULE.bazel <<EOF
new_git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'new_git_repository')
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
    cat >> MODULE.bazel <<EOF
new_git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'new_git_repository')
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
cat ../+_repo_rules+pluto/info
EOF
  chmod +x planets/planet_info.sh

  bazel run //planets:planet-info >& $TEST_log \
      || echo "Expected build/run to succeed"
  if [ "$1" == "0-initial" ]; then
      expect_log "Pluto is a planet"
  else
      expect_log "Pluto is a dwarf planet"
  fi

  git_repos_count=$(find $(bazel info output_base)/external/+_repo_rules+pluto -type d -name .git | wc -l)
  assert_equals $git_repos_count 0
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
#   MODULE.bazel
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
  cat >> MODULE.bazel <<EOF
new_git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'new_git_repository')
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
cat ../+_repo_rules+outer_planets/neptune/info
cat ../+_repo_rules+outer_planets/pluto/info
EOF
  chmod +x planets/planet_info.sh

  bazel run //planets:planet-info >& $TEST_log \
    || echo "Expected build/run to succeed"
  expect_log "Neptune is a planet"
  expect_log "Pluto is a planet"
}

function test_new_git_repository_submodules_with_recursive_init_modules() {
  local outer_planets_repo_dir=$TEST_TMPDIR/repos/outer-planets

  # Create a workspace that clones the outer_planets repository.
  cat >> MODULE.bazel <<EOF
new_git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'new_git_repository')
new_git_repository(
    name = "outer_planets",
    remote = "$outer_planets_repo_dir",
    tag = "1-submodule",
    recursive_init_submodules = 1,
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
cat ../+_repo_rules+outer_planets/neptune/info
cat ../+_repo_rules+outer_planets/pluto/info
EOF
  chmod +x planets/planet_info.sh

  bazel run //planets:planet-info >& $TEST_log \
    || echo "Expected build/run to succeed"
  expect_log "Neptune is a planet"
  expect_log "Pluto is a planet"
}

function test_git_repository_not_refetched_on_server_restart() {
  local repo_dir=$TEST_TMPDIR/repos/refetch

  cat >> MODULE.bazel <<EOF
git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='22095302abaf776886879efa5129aa4d44c53017', verbose=True)
EOF

  # Use batch to force server restarts.
  bazel --batch build @g//:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 1" bazel-genfiles/external/+_repo_rules+g/go

  # Without changing anything, restart the server, which should not cause the checkout to be re-cloned.
  bazel --batch build @g//:g >& $TEST_log || fail "Build failed"
  expect_not_log "Cloning"
  assert_contains "GIT 1" bazel-genfiles/external/+_repo_rules+g/go

  # Change the commit id, which should cause the checkout to be re-cloned.
  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='db134ae9b644d8237954a8e6f1ef80fcfd85d521', verbose=True)
EOF

  bazel --batch build @g//:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 2" bazel-genfiles/external/+_repo_rules+g/go

  # Change the MODULE.bazel but not the commit id, which should not cause the checkout to be re-cloned.
  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
# This comment line is to change the line numbers, which should not cause Bazel
# to refetch the repository
git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='db134ae9b644d8237954a8e6f1ef80fcfd85d521', verbose=True)
EOF

  bazel --batch build @g//:g >& $TEST_log || fail "Build failed"
  expect_not_log "Cloning"
  assert_contains "GIT 2" bazel-genfiles/external/+_repo_rules+g/go
}

function test_git_repository_not_refetched_on_server_restart_strip_prefix() {
  local repo_dir=$TEST_TMPDIR/repos/refetch
  # Change the strip_prefix which should cause a new checkout
  cat >> MODULE.bazel <<EOF
git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='17ea13b242e4cbcc27a6ef745939ebb7dcccea10', verbose=True)
EOF
  bazel --batch build @g//gdir:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 2" bazel-genfiles/external/+_repo_rules+g/gdir/go

  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='17ea13b242e4cbcc27a6ef745939ebb7dcccea10', verbose=True, strip_prefix="gdir")
EOF
  bazel --batch build @g//:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 2" bazel-genfiles/external/+_repo_rules+g/go
}


function test_git_repository_refetched_when_commit_changes() {
  local repo_dir=$TEST_TMPDIR/repos/refetch

  cat >> MODULE.bazel <<EOF
git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='22095302abaf776886879efa5129aa4d44c53017', verbose=True)
EOF

  bazel build @g//:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 1" bazel-genfiles/external/+_repo_rules+g/go

  # Change the commit id, which should cause the checkout to be re-cloned.
  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='db134ae9b644d8237954a8e6f1ef80fcfd85d521', verbose=True)
EOF

  bazel build @g//:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 2" bazel-genfiles/external/+_repo_rules+g/go
}

function test_git_repository_and_nofetch() {
  local repo_dir=$TEST_TMPDIR/repos/refetch

  cat >> MODULE.bazel <<EOF
git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='22095302abaf776886879efa5129aa4d44c53017')
EOF

  bazel build --nofetch @g//:g >& $TEST_log && fail "Build succeeded"
  expect_log "fetching repositories is disabled"
  bazel build @g//:g >& $TEST_log || fail "Build failed"
  assert_contains "GIT 1" bazel-genfiles/external/+_repo_rules+g/go

  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='db134ae9b644d8237954a8e6f1ef80fcfd85d521')
EOF

  bazel build --nofetch @g//:g >& $TEST_log || fail "Build failed"
  expect_log "External repository '+_repo_rules+g' is not up-to-date"
  assert_contains "GIT 1" bazel-genfiles/external/+_repo_rules+g/go
  bazel build  @g//:g >& $TEST_log || fail "Build failed"
  assert_contains "GIT 2" bazel-genfiles/external/+_repo_rules+g/go

  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
git_repository(name='g', remote='$repo_dir', commit='17ea13b242e4cbcc27a6ef745939ebb7dcccea10', strip_prefix="gdir")
EOF

  bazel build --nofetch @g//:g >& $TEST_log || fail "Build failed"
  expect_log "External repository '+_repo_rules+g' is not up-to-date"
  bazel build  @g//:g >& $TEST_log || fail "Build failed"
  assert_contains "GIT 2" bazel-genfiles/external/+_repo_rules+g/go

}

# Helper function for setting up the workspace as follows
#
# $WORKSPACE_DIR/
#   MODULE.bazel
#   planets/
#     planet_info.sh
#     BUILD
function setup_error_test() {
  mkdir -p planets
  cat > planets/planet_info.sh <<EOF
#!/bin/sh
cat external/+_repo_rules+pluto/info
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
#   MODULE.bazel
#   BUILD
#   info
function test_git_repository_both_commit_tag_error() {
  setup_error_test
  local pluto_repo_dir=$(get_pluto_repo)
  # Commit corresponds to tag 1-build. See testdata/pluto.git_log.
  local commit_hash="52f9a3f87a2dd17ae0e5847bbae9734f09354afd"

  cat >> MODULE.bazel <<EOF
git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
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
#   MODULE.bazel
#   BUILD
#   info
function test_git_repository_no_commit_tag_error() {
  setup_error_test
  local pluto_repo_dir=$(get_pluto_repo)

  cat >> MODULE.bazel <<EOF
git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
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
  local pluto_repo_dir=$(get_pluto_repo)

  cat >> MODULE.bazel <<EOF
git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
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
  local pluto_repo_dir=$(get_pluto_repo)

  cat >> MODULE.bazel <<EOF
git_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')
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

# Verifies that load statement works while using strip_prefix.
#
# This test uses the strip-prefix Git repository, which contains the
# following files:
#
# strip-prefix
# └── prefix-foo
#     ├── BUILD
#     ├── MODULE.bazel
#     └── defs.bzl
function test_git_repository_with_strip_prefix_for_load_statement() {
  setup_error_test
  local strip_prefix_repo_dir=$TEST_TMPDIR/repos/strip-prefix

  cat >> MODULE.bazel <<EOF
git_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
git_repository(
    name = "foo",
    remote = "$strip_prefix_repo_dir",
    commit = "f8167a60de4460e89601724fb13b4fc505da3f3d",
    strip_prefix = "prefix-foo",
)
EOF

  cat > BUILD <<EOF
load("@foo//:defs.bzl", "FOO")
EOF

  bazel build //:all >& $TEST_log || fail "Expect bazel build to succeed."
}

run_suite "Starlark git_repository tests"
