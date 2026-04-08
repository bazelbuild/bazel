#!/usr/bin/env bash
#
# Copyright 2026 The Bazel Authors. All rights reserved.
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
# Test mercurial_repository rules.
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

if is_windows; then
  # Enable symlink runfiles tree to make bazel run work
  add_to_bazelrc "build --enable_runfiles"
fi

# Global test setup.
#
# Creates the Mercurial repositories in the test temporary directory.
function set_up() {
  if ! command -v hg &> /dev/null; then
      echo "Error: hg is not installed."
      exit 1
  fi

  local repos_dir=$TEST_TMPDIR/repos
  if [ -e "$repos_dir" ]; then
    rm -rf $repos_dir
  fi

  mkdir -p $repos_dir
  cd $repos_dir

  # pluto
  mkdir pluto
  cd pluto
  $(rlocation io_bazel/src/test/shell/bazel/setup_hg_pluto_repo.sh)
  cd ..

  # refetch
  mkdir refetch
  cd refetch
  $(rlocation io_bazel/src/test/shell/bazel/setup_hg_refetch_repo.sh)
  cd ..

  # strip-prefix
  mkdir strip-prefix
  cd strip-refix
  $(rlocation io_bazel/src/test/shell/bazel/setup_hg_strip_prefix_repo.sh)
  cd ..

  setup_module_dot_bazel

  # Fix environment variables for a hermetic use of hg.
  export HGRCPATH=
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

# Test cloning a Mercurial repository using the mercurial_repository rule.
#
# This test uses the pluto Mercurial repository at tag 1-build, which contains the
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
# //planets has a dependency on a target in the pluto Mercurial repository.
function do_mercurial_repository_test() {
  local pluto_repo_dir=$(get_pluto_repo)
  # Commit corresponds to tag 1-build. See pluto_repo.hg.log.
  local commit_hash="$1"
  local strip_prefix=""
  [ $# -eq 2 ] && strip_prefix="strip_prefix=\"$2\","

  # Create a workspace that clones the repository at the first commit.
  cat >> MODULE.bazel <<EOF
mercurial_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:hg.bzl', 'mercurial_repository')
mercurial_repository(
    name = "pluto",
    remote = "$pluto_repo_dir",
    revision = "$commit_hash",
    $strip_prefix
)
EOF
  add_rules_shell "MODULE.bazel"
  mkdir -p planets
  cat > planets/BUILD <<EOF
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

sh_binary(
    name = "planet-info",
    srcs = ["planet_info.sh"],
    data = ["@pluto//:pluto"],
)
EOF

  cat > planets/planet_info.sh <<EOF
#!/bin/sh
cat ../+mercurial_repository+pluto/info
EOF
  chmod +x planets/planet_info.sh

  bazel run //planets:planet-info >& $TEST_log \
    || echo "Expected build/run to succeed"
  expect_log "Pluto is a dwarf planet"

  hg_repos_count=$(find $(bazel info output_base)/external/+mercurial_repository+pluto -type d -name .hg | wc -l)
  assert_equals $hg_repos_count 0
}

function test_mercurial_repository() {
  do_mercurial_repository_test "baa0483a956434b5a104147d52de87eb03cb5654"
}

function test_mercurial_repository_strip_prefix() {
  # This commit has the files in a subdirectory named 'pluto'
  # so we strip_prefix and the build should still work.
  do_mercurial_repository_test "a5c5a6bc585fe5a906aaf40a72f87b9835cfbbc9" "pluto"
}

function test_mercurial_repository_with_build_file() {
  do_mercurial_repository_test_with_build "0-initial" "build_file"
}

function test_mercurial_repository_with_build_file_strip_prefix() {
  do_mercurial_repository_test_with_build "3-subdir-bare" "build_file" "pluto"
}

function test_mercurial_repository_with_build_file_strip_prefix_default_branch() {
  do_mercurial_repository_test_with_build "" "build_file" "pluto"
}

function test_mercurial_repository_with_build_file_content() {
  do_mercurial_repository_test_with_build "0-initial" "build_file_content"
}

function test_mercurial_repository_with_build_file_content_strip_prefix() {
  do_mercurial_repository_test_with_build "3-subdir-bare" "build_file_content" "pluto"
}

function test_mercurial_repository_with_build_file_content_strip_prefix_default_branch() {
  do_mercurial_repository_test_with_build "" "build_file_content" "pluto"
}

# Test cloning a Mercurial repository using the mercurial_repository rule.
#
# This test uses the pluto Mercurial repository at tag 0-initial, which contains the
# following files:
#
# pluto/
#   info
#
# Then it uses the pluto Mercurial repository at tag 3-subdir-bare, which contains the
# following files:
# pluto/
#   pluto/
#     info
#
# Finally, it uses the pluto Mercurial repository at the default branch, which is the
# master branch, which is at the same revision as the 3-subdir-bare tag.
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
# //planets has a dependency on a target in the $TEST_TMPDIR/pluto Mercurial
# repository.
function do_mercurial_repository_test_with_build() {
  local pluto_repo_dir=$(get_pluto_repo)
  local strip_prefix=""
  local tag=""
  [ $# -eq 3 ] && strip_prefix="strip_prefix=\"$3\","
  [ "$1" != "" ] && tag="revision = \"$1\","

  # Create a workspace that clones the repository at the first commit.

  if [ "$2" == "build_file" ]; then
    cat >> MODULE.bazel <<EOF
mercurial_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:hg.bzl', 'mercurial_repository')
mercurial_repository(
    name = "pluto",
    remote = "$pluto_repo_dir",
    $tag
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
mercurial_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:hg.bzl', 'mercurial_repository')
mercurial_repository(
    name = "pluto",
    remote = "$pluto_repo_dir",
    $tag
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
  add_rules_shell "MODULE.bazel"

  mkdir -p planets
  cat > planets/BUILD <<EOF
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

sh_binary(
    name = "planet-info",
    srcs = ["planet_info.sh"],
    data = ["@pluto//:pluto"],
)
EOF

  cat > planets/planet_info.sh <<EOF
#!/bin/sh
cat ../+mercurial_repository+pluto/info
EOF
  chmod +x planets/planet_info.sh

  bazel run //planets:planet-info >& $TEST_log \
      || echo "Expected build/run to succeed"
  if [ "$1" == "0-initial" ]; then
      expect_log "Pluto is a planet"
  else
      expect_log "Pluto is a dwarf planet"
  fi

  hg_repos_count=$(find $(bazel info output_base)/external/+mercurial_repository+pluto -type d -name .hg | wc -l)
  assert_equals $hg_repos_count 0
}

function test_mercurial_repository_not_refetched_on_server_restart() {
  # Testing refetch behavior, so disable the repo contents cache
  add_to_bazelrc "common --repo_contents_cache="
  local repo_dir=$TEST_TMPDIR/repos/refetch

  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
mercurial_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:hg.bzl', 'mercurial_repository')
mercurial_repository(name='g', remote='$repo_dir', revision='92872ece5144313d5f29000322ea77c9a7d7159a', verbose=True)
EOF

  # Use batch to force server restarts.
  bazel --batch build @g//:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 1" bazel-genfiles/external/+mercurial_repository+g/go

  # Without changing anything, restart the server, which should not cause the checkout to be re-cloned.
  bazel --batch build @g//:g >& $TEST_log || fail "Build failed"
  expect_not_log "Cloning"
  assert_contains "GIT 1" bazel-genfiles/external/+mercurial_repository+g/go

  # Change the commit id, which should cause the checkout to be re-cloned.
  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
mercurial_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:hg.bzl', 'mercurial_repository')
mercurial_repository(name='g', remote='$repo_dir', revision='e34c5d9fe274c05d6ff499d249a5461770b8be8e', verbose=True)
EOF

  bazel --batch build @g//:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 2" bazel-genfiles/external/+mercurial_repository+g/go

  # Change the MODULE.bazel but not the commit id, which should not cause the checkout to be re-cloned.
  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
# This comment line is to change the line numbers, which should not cause Bazel
# to refetch the repository
mercurial_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:hg.bzl', 'mercurial_repository')
mercurial_repository(name='g', remote='$repo_dir', revision='e34c5d9fe274c05d6ff499d249a5461770b8be8e', verbose=True)
EOF

  bazel --batch build @g//:g >& $TEST_log || fail "Build failed"
  expect_not_log "Cloning"
  assert_contains "GIT 2" bazel-genfiles/external/+mercurial_repository+g/go
}

function test_mercurial_repository_not_refetched_on_server_restart_strip_prefix() {
  # Testing refetch behavior, so disable the repo contents cache
  add_to_bazelrc "common --repo_contents_cache="
  local repo_dir=$TEST_TMPDIR/repos/refetch
  # Change the strip_prefix which should cause a new checkout
  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
mercurial_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:hg.bzl', 'mercurial_repository')
mercurial_repository(name='g', remote='$repo_dir', revision='ddfce8df599ad91e9f3416ed1a89f60321fa3f05', verbose=True)
EOF
  bazel --batch build @g//gdir:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 2" bazel-genfiles/external/+mercurial_repository+g/gdir/go

  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
mercurial_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:hg.bzl', 'mercurial_repository')
mercurial_repository(name='g', remote='$repo_dir', revision='ddfce8df599ad91e9f3416ed1a89f60321fa3f05', verbose=True, strip_prefix="gdir")
EOF
  bazel --batch build @g//:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 2" bazel-genfiles/external/+mercurial_repository+g/go
}


function test_mercurial_repository_refetched_when_commit_changes() {
  # Testing refetch behavior, so disable the repo contents cache
  add_to_bazelrc "common --repo_contents_cache="
  local repo_dir=$TEST_TMPDIR/repos/refetch

  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
mercurial_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:hg.bzl', 'mercurial_repository')
mercurial_repository(name='g', remote='$repo_dir', revision='92872ece5144313d5f29000322ea77c9a7d7159a', verbose=True)
EOF

  bazel build @g//:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 1" bazel-genfiles/external/+mercurial_repository+g/go

  # Change the commit id, which should cause the checkout to be re-cloned.
  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
mercurial_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:hg.bzl', 'mercurial_repository')
mercurial_repository(name='g', remote='$repo_dir', revision='e34c5d9fe274c05d6ff499d249a5461770b8be8e', verbose=True)
EOF

  bazel build @g//:g >& $TEST_log || fail "Build failed"
  expect_log "Cloning"
  assert_contains "GIT 2" bazel-genfiles/external/+mercurial_repository+g/go
}

function test_mercurial_repository_and_nofetch() {
  # Testing refetch behavior, so disable the repo contents cache
  add_to_bazelrc "common --repo_contents_cache="
  local repo_dir=$TEST_TMPDIR/repos/refetch

  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
mercurial_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:hg.bzl', 'mercurial_repository')
mercurial_repository(name='g', remote='$repo_dir', revision='92872ece5144313d5f29000322ea77c9a7d7159a')
EOF

  bazel build --nofetch @g//:g >& $TEST_log && fail "Build succeeded"
  expect_log "fetching repositories is disabled"
  bazel build @g//:g >& $TEST_log || fail "Build failed"
  assert_contains "GIT 1" bazel-genfiles/external/+mercurial_repository+g/go

  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
mercurial_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:hg.bzl', 'mercurial_repository')
mercurial_repository(name='g', remote='$repo_dir', revision='e34c5d9fe274c05d6ff499d249a5461770b8be8e')
EOF

  bazel build --nofetch @g//:g >& $TEST_log || fail "Build failed"
  expect_log "External repository '@@+mercurial_repository+g' is not up-to-date"
  assert_contains "GIT 1" bazel-genfiles/external/+mercurial_repository+g/go
  bazel build  @g//:g >& $TEST_log || fail "Build failed"
  assert_contains "GIT 2" bazel-genfiles/external/+mercurial_repository+g/go

  rm MODULE.bazel
  cat >> MODULE.bazel <<EOF
mercurial_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:hg.bzl', 'mercurial_repository')
mercurial_repository(name='g', remote='$repo_dir', revision='ddfce8df599ad91e9f3416ed1a89f60321fa3f05', strip_prefix="gdir")
EOF

  bazel build --nofetch @g//:g >& $TEST_log || fail "Build failed"
  expect_log "External repository '@@+mercurial_repository+g' is not up-to-date"
  bazel build  @g//:g >& $TEST_log || fail "Build failed"
  assert_contains "GIT 2" bazel-genfiles/external/+mercurial_repository+g/go

}

# Helper function for setting up the workspace as follows
#
# $WORKSPACE_DIR/
#   MODULE.bazel
#   planets/
#     planet_info.sh
#     BUILD
function setup_error_test() {
  add_rules_shell "MODULE.bazel"
  mkdir -p planets
  cat > planets/planet_info.sh <<EOF
#!/bin/sh
cat external/+mercurial_repository+pluto/info
EOF

  cat > planets/BUILD <<EOF
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

sh_binary(
    name = "planet-info",
    srcs = ["planet_info.sh"],
    data = ["@pluto//:pluto"],
)
EOF
}

# Verifies that if a non-existent subdirectory is supplied, then strip_prefix
# throws an error.
function test_invalid_strip_prefix_error() {
  setup_error_test
  local pluto_repo_dir=$(get_pluto_repo)

  cat >> MODULE.bazel <<EOF
mercurial_repository = use_repo_rule('@bazel_tools//tools/build_defs/repo:hg.bzl', 'mercurial_repository')
mercurial_repository(
    name = "pluto",
    remote = "$pluto_repo_dir",
    revision = "1-build",
    strip_prefix = "dir_does_not_exist"
)
EOF

  bazel fetch //planets:planet-info >& $TEST_log \
    || echo "Expect run to fail."
  expect_log "strip_prefix at dir_does_not_exist does not exist in repo"
}

# Verifies that load statement works while using strip_prefix.
#
# This test uses the strip-prefix Mercurial repository, which contains the
# following files:
#
# strip-prefix
# └── prefix-foo
#     ├── BUILD
#     ├── MODULE.bazel
#     └── defs.bzl
function test_hg_repository_with_strip_prefix_for_load_statement() {
  setup_error_test
  local strip_prefix_repo_dir=$TEST_TMPDIR/repos/strip-prefix

  cat >> MODULE.bazel <<EOF
mercurial_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:hg.bzl", "mercurial_repository")
mercurial_repository(
    name = "foo",
    remote = "$strip_prefix_repo_dir",
    revision = "ee31cd98d80d6c9b6a2e1dd0fdfeb27538da3163",
    strip_prefix = "prefix-foo",
)
EOF

  cat > BUILD <<EOF
load("@foo//:defs.bzl", "FOO")
EOF

  bazel build //:all >& $TEST_log || fail "Expect bazel build to succeed."
}

run_suite "Starlark mercurial_repository tests"
