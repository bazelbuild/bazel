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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# The clash canary flags are built into the canonicalize-flags command
# specifically for this test suite.
canary_clash_error="option '--flag_clash_canary' was expanded to from both "
canary_clash_error+="option '--flag_clash_canary_expander1' and "
canary_clash_error+="option '--flag_clash_canary_expander2'."

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
