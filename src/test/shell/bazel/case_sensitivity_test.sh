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

# --- begin runfiles.bash initialization v2 ---
# Copy-pasted from the Bazel Bash runfiles library v2.
set -uo pipefail; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v2 ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  declare -r is_windows=true
  declare -r is_mac=false
  ;;
darwin*)
  declare -r is_windows=false
  declare -r is_mac=true
  ;;
*)
  declare -r is_windows=false
  declare -r is_mac=false
  ;;
esac

if $is_windows; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

# ------------------------------------------------------------------------------
# TESTS
# ------------------------------------------------------------------------------

# Assert that repository names are case sensitive.
# Regression test for https://github.com/bazelbuild/bazel/issues/9216
function test_case_sensitive_repository_name() {
  local -r root="${FUNCNAME[0]}" 
  mkdir -p "$root/MyRepo" || fail "mkdir -p $root/MyRepo"
  mkdir -p "$root/MyMain" || fail "mkdir -p $root/MyMain"

  touch "$root/MyRepo/WORKSPACE"
  echo "filegroup(name = 'x')" > "$root/MyRepo/BUILD"

  if $is_windows || $is_mac; then
    # As of 2019-10-01, local_repository.path is case-ignoring on Windows and
    # macOS. This is a bug and should be fixed, and this test should be updated.
    # Now this test just guards the current behavior.
    echo "local_repository(name = 'MyRepo', path = '../myrepo')" \
        > "$root/MyMain/WORKSPACE"
  else
    echo "local_repository(name = 'MyRepo', path = '../MyRepo')" \
        > "$root/MyMain/WORKSPACE"
  fi

  cd "$root/MyMain"
  bazel query @myrepo//:all >&"$TEST_log" && fail "Expected failure" || true
  expect_log "'@myrepo' could not be resolved"

  bazel query @MyRepo//:all >&"$TEST_log" || fail "Expected success"
  expect_not_log "could not be resolved"
  expect_log "@MyRepo//:x"

  bazel query @myrepo//:all >&"$TEST_log" && fail "Expected failure" || true
  bazel shutdown
  expect_log "'@myrepo' could not be resolved"
}

run_suite "git_repository tests"
