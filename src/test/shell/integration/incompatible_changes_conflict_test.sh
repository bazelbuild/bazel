#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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

# Tests that there are no conflicts among all of the flags in the transitive
# expansion of --all_incompatible_changes. This is an integration test because
# it is difficult to know in a unit test exactly what features (OptionsBase
# subclasses) are passed to the parser from within a unit test.

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
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

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  # As of 2018-08-14, Bazel on Windows only supports MSYS Bash.
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  # Disable MSYS path conversion that converts path-looking command arguments to
  # Windows paths (even if they arguments are not in fact paths).
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

# The clash canary flags are built into the canonicalize-flags command
# specifically for this test suite.
canary_clash_error="option '--flag_clash_canary' was expanded to from both "
canary_clash_error+="option '--flag_clash_canary_expander1' and "
canary_clash_error+="option '--flag_clash_canary_expander2'"

# Ensures that we didn't change the formatting of the warning message or
# disable the warning.
function test_conflict_warning_is_working() {
  bazel canonicalize-flags --show_warnings -- \
    --flag_clash_canary_expander1 --flag_clash_canary_expander2 \
    &>$TEST_log || fail "bazel canonicalize-flags failed";
  fail_msg="Did not find expected flag conflict warning"
  expect_log "$canary_clash_error" "$fail_msg"
}

# Ensures that canonicalize-flags doesn't emit warnings unless requested.
function test_canonicalize_flags_suppresses_warnings() {
  bazel canonicalize-flags -- \
    --flag_clash_canary_expander1 --flag_clash_canary_expander2 \
    &>$TEST_log || fail "bazel canonicalize-flags failed";
  fail_msg="canonicalize-flags should have suppressed parser warnings since "
  fail_msg+="--show_warnings wasn't specified"
  expect_not_log "$canary_clash_error" "$fail_msg"
}

function test_no_conflicts_among_incompatible_changes() {
  bazel canonicalize-flags --show_warnings -- --all_incompatible_changes \
    &>$TEST_log || fail "bazel canonicalize-flags failed";
  expected="The option '.*' was expanded to from both option "
  expected+="'.*' and option '.*'."
  fail_msg="Options conflict in expansion of --all_incompatible_changes"
  expect_not_log "$expected" "$fail_msg"
}


run_suite "incompatible_changes_conflict_test"
