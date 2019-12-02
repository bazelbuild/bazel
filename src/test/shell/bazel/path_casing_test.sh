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

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
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

if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

#### TESTS #############################################################

function test_case_sensitive_pkg_names() {
  local -r pkg1="Pkg${RANDOM}"
  local -r pkg2="MyPackage"
  mkdir -p "$pkg1/$pkg2" || fail "Could not mkdir $pkg1/$pkg2"
  echo "filegroup(name = 'MyTarget')" > "$pkg1/$pkg2/BUILD"

  local -r pkg1_upper=$(echo "$pkg1" | tr '[:lower:]' '[:upper:]')
  local -r pkg2_upper=$(echo "$pkg2" | tr '[:lower:]' '[:upper:]')

  [[ "$pkg1_upper" != "$pkg1" ]] || fail "Expected strings ($pkg1) to differ"
  [[ "$pkg2_upper" != "$pkg2" ]] || fail "Expected strings ($pkg2) to differ"

  if $is_windows || $is_mac; then
    [[ -e "$pkg1_upper/$pkg2_upper/BUILD" ]] \
        || fail "Expected case-insensitive semantics"
  else
    if [[ -e "$pkg1_upper/$pkg2_upper/BUILD" ]]; then
      fail "Expected case-sensitive semantics"
    fi
  fi

  # The wrong casing should not work.
  if bazel --experimental_check_label_casing query \
      //$pkg1_upper/$pkg2_upper:all >&"$TEST_log"; then
    fail "Expected failure"
  fi
  if $is_windows || $is_mac; then
    expect_log "no such package.*$pkg1_upper/$pkg2_upper.*path casing is wrong"
  else
    expect_log "no such package.*$pkg1_upper/$pkg2_upper.*BUILD file not found"
  fi

  if bazel --experimental_check_label_casing query \
      //$pkg1/$pkg2_upper:all >&"$TEST_log"; then
    fail "Expected failure"
  fi
  if $is_windows || $is_mac; then
    expect_log "no such package.*$pkg1/$pkg2_upper.*path casing is wrong"
  else
    expect_log "no such package.*$pkg1/$pkg2_upper.*BUILD file not found"
  fi

  if bazel --experimental_check_label_casing query \
      //$pkg1_upper/$pkg2:all >&"$TEST_log"; then
    fail "Expected failure"
  fi
  if $is_windows || $is_mac; then
    expect_log "no such package.*$pkg1_upper/$pkg2.*path casing is wrong"
  else
    expect_log "no such package.*$pkg1_upper/$pkg2.*BUILD file not found"
  fi

  # The right casing should work.
  bazel --experimental_check_label_casing query \
      //$pkg1/$pkg2:all >&"$TEST_log" || fail "Expected success"
  expect_not_log "no such package"
  expect_log "//$pkg1/$pkg2:MyTarget"

  # The wrong casing should still not work.
  if bazel --experimental_check_label_casing query \
      //$pkg1_upper/$pkg2_upper:all >&"$TEST_log"; then
    fail "Expected failure"
  fi
  if $is_windows || $is_mac; then
    expect_log "no such package.*$pkg1_upper/$pkg2_upper.*path casing is wrong"
  else
    expect_log "no such package.*$pkg1_upper/$pkg2_upper.*BUILD file not found"
  fi

  # The wrong casing should only work on Windows and macOS, and when case
  # checking is disabled.
  if $is_windows || $is_mac; then
    bazel --noexperimental_check_label_casing query \
        //$pkg1_upper/$pkg2_upper:all >&"$TEST_log" || fail "Expected success"
  fi
}

run_suite "Path casing validation tests"
