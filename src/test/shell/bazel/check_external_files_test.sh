#!/bin/bash
#
# Copyright 2019 The Bazel Authors. All rights reserved.
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
# Verify that archives can be unpacked, even if they contain strangely named
# files.

# --- begin runfiles.bash initialization ---
set -euo pipefail
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

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

disable_bzlmod

function get_extrepourl() {
  if $is_windows; then
    echo "file:///$(cygpath -m $1)"
  else
    echo "file://$1"
  fi
}

setup_remote() {
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir remote
  (
    cd remote
    echo 'genrule(name="g", outs=["go"], cmd="echo GO > $@")' > BUILD
  )
  tar cvf remote.tar remote
  rm -rf remote

  mkdir main
  cd main
  cat >> "$(create_workspace_with_default_repos WORKSPACE)" <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="remote",
  strip_prefix="remote",
  urls=["$(get_extrepourl $WRKDIR)/remote.tar"],
)
EOF
}

setup_local() {
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir local_rep
  (
    cd local_rep
    create_workspace_with_default_repos WORKSPACE
    echo 'genrule(name="g", outs=["go"], cmd="echo GO > $@")' > BUILD
  )

  mkdir main
  cd main
  cat >> "$(create_workspace_with_default_repos WORKSPACE)" <<EOF
load("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(
  name="local_rep",
  path="../local_rep",
)
EOF
}

test_check_external_files() {
  setup_remote
  bazel build @remote//:g >& "$TEST_log" || fail "Expected build to succeed"

  echo "broken file" > bazel-main/external/remote/BUILD
  # The --noexperimental_check_external_repository_files flag doesn't notice the file is broken
  bazel build --noexperimental_check_external_repository_files @remote//:g >& "$TEST_log" || fail "Expected build to succeed"

  bazel build @remote//:g >& "$TEST_log" && fail "Expected build to fail" || true
  expect_log "no such target '@@remote//:g'"
}

test_check_all_flags_fast() {
  setup_remote
  msg="About to scan skyframe graph checking for filesystem nodes"

  bazel build --watchfs @remote//:g >& "$TEST_log" || fail "Expected build to succeed"
  instances=$(grep -c "$msg" "$(bazel info server_log)")
  [[ $instances -eq 1 ]] || fail "Should have only been 1 instance, got $instances"

  echo "broken file" > bazel-main/external/remote/BUILD

  bazel build \
    --noexperimental_check_external_repository_files \
    --noexperimental_check_output_files \
    --watchfs \
    @remote//:g >& "$TEST_log" || fail "Expected build to succeed"

  instances=$(grep -c "$msg" "$(bazel info server_log)")
  [[ $instances -eq 1 ]] || fail "Should have only been 1 instance (from the first build), got $instances"
}

run_local_repository_isnt_affected() {
  local -r extra_args="$1"
  shift

  setup_local
  bazel build @local_rep//:g >& "$TEST_log" || fail "Expected build to succeed"

  echo "broken file" > ../local_rep/BUILD
  # The --noexperimental_check_external_repository_files flag still notices the file is broken
  bazel build \
    --noexperimental_check_external_repository_files \
    $extra_args \
    @local_rep//:g >& "$TEST_log" && fail "Expected build to fail" || true
  bazel build --noexperimental_check_external_repository_files @local_rep//:g >& "$TEST_log" && fail "Expected build to fail" || true
  expect_log "no such target '@@local_rep//:g'"
}

test_local_repository_isnt_affected() {
  run_local_repository_isnt_affected "--nowatchfs"
}

test_local_repository_isnt_affected_with_skips() {
  run_local_repository_isnt_affected "--noexperimental_check_output_files --watchfs"
}

run_override_repository_isnt_affected() {
  local -r extra_args="$1"
  shift

  setup_local
  create_workspace_with_default_repos WORKSPACE
  bazel build @local_rep//:g >& "$TEST_log" && fail "Expected build to fail" || true
  expect_log "no such package '@@local_rep//'"

  argv="--override_repository=local_rep=$(pwd)/../local_rep"
  bazel build "$argv" $extra_args @local_rep//:g >& "$TEST_log" || fail "Expected build to succeed"

  echo "broken file" > ../local_rep/BUILD
  # The --noexperimental_check_external_repository_files flag still notices the file is broken
  bazel build \
    --noexperimental_check_external_repository_files \
    "$argv" \
    $extra_args \
    @local_rep//:g >& "$TEST_log" && fail "Expected build to fail" || true
  expect_log "no such target '@@local_rep//:g'"
}

test_override_repository_isnt_affected() {
  run_override_repository_isnt_affected "--nowatchfs"
}
test_override_repository_isnt_affected_with_skips() {
  run_override_repository_isnt_affected "--noexperimental_check_output_files --watchfs"
}

test_no_fetch_then_fetch() {
  setup_remote
  bazel build \
    --nofetch \
    --noexperimental_check_external_repository_files \
    --noexperimental_check_output_files \
    --watchfs \
    @remote//:g >& "$TEST_log" && fail "Expected build to fail" || true
  expect_log "no such package"
  expect_log "fetching repositories is disabled"
  bazel build \
    --fetch \
    --noexperimental_check_external_repository_files \
    --noexperimental_check_output_files \
    --watchfs \
    @remote//:g >& "$TEST_log" || fail "Expected build to pass"
}

test_no_build_doesnt_break_the_cache() {
  setup_remote
  bazel build \
    --nobuild \
    --noexperimental_check_external_repository_files \
    --noexperimental_check_output_files \
    --watchfs \
    @remote//:g >& "$TEST_log" || fail "Expected build to pass"
  [[ ! -f bazel-main/external/remote/BUILD ]] || fail "external files shouldn't have been made"
  bazel build \
    --noexperimental_check_external_repository_files \
    --noexperimental_check_output_files \
    --watchfs \
    @remote//:g >& "$TEST_log" || fail "Expected build to pass"
}

test_symlink_outside_still_checked() {
  mkdir main
  cd main
  create_workspace_with_default_repos WORKSPACE
  echo 'sh_test(name = "symlink", srcs = ["symlink.sh"])' > BUILD

  mkdir ../foo
  echo 'exit 0' > ../foo/foo.sh
  chmod u+x ../foo/foo.sh
  ln -s ../foo/foo.sh symlink.sh

  bazel test \
    --noexperimental_check_external_repository_files \
    --noexperimental_check_output_files \
    --watchfs \
    :symlink >& "$TEST_log" || fail "Expected build to succeed"

  bazel test \
    --noexperimental_check_external_repository_files \
    --noexperimental_check_output_files \
    --watchfs \
    :symlink >& "$TEST_log" || fail "Expected build to succeed"
  expect_log '//:symlink.*cached'

  echo 'exit 1' > ../foo/foo.sh

  bazel test \
    --noexperimental_check_external_repository_files \
    --noexperimental_check_output_files \
    --watchfs \
    :symlink >& "$TEST_log" && fail "Expected build to fail" || true
  expect_not_log '//:symlink.*cached'
  expect_log '1 test FAILED'
}

run_suite "check_external_files tests"
