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
# Integration tests for the visibility checking of the tests referenced by a
# test_suite.
#
# Regression tests for https://github.com/bazelbuild/bazel/issues/14053: with
# --expand_test_suites (the default for build/test/coverage), test_suite targets
# are replaced by their constituent tests before analysis. The suite itself was
# never analyzed, so the visibility of its `tests` attribute references was
# never checked, unlike with cquery (which sets --noexpand_test_suites).
#
# Expanded test suites are now analyzed too, but they must neither be built nor
# show up as top-level targets (build summary, BEP, ...).

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

# The analysis of expanded test suites is wired differently with and without
# merged analysis/execution (Skymeld), so most scenarios run in both modes.
readonly SKYMELD="--experimental_merged_skyframe_analysis_execution"
readonly NO_SKYMELD="--noexperimental_merged_skyframe_analysis_execution"

#### HELPERS ###############################################################

function write_test() {
  # Writes //$1:$2, a trivially passing sh_test with the given visibility.
  local -r pkg="$1"
  local -r name="$2"
  local -r visibility="$3"

  mkdir -p "${pkg}"
  if [[ ! -f "${pkg}/BUILD" ]]; then
    echo 'load("@rules_shell//shell:sh_test.bzl", "sh_test")' > "${pkg}/BUILD"
  fi
  cat >> "${pkg}/BUILD" <<EOF
sh_test(
    name = "${name}",
    srcs = ["${name}.sh"],
    visibility = ${visibility},
)
EOF
  cat > "${pkg}/${name}.sh" <<'EOF'
#!/bin/sh
exit 0
EOF
  chmod +x "${pkg}/${name}.sh"
}

function write_suite() {
  # Writes //$1:$2, a test_suite referencing the given labels.
  local -r pkg="$1"
  local -r name="$2"
  shift 2

  mkdir -p "${pkg}"
  {
    echo "test_suite("
    echo "    name = \"${name}\","
    echo "    tests = ["
    for label in "$@"; do
      echo "        \"${label}\","
    done
    echo "    ],"
    echo "    visibility = [\"//visibility:public\"],"
    echo ")"
  } >> "${pkg}/BUILD"
}

function write_test_and_suite() {
  # Creates two packages:
  #   //$1/testpkg:the_test   an sh_test with the given visibility
  #   //$1/suitepkg:the_suite a test_suite referencing the_test
  local -r prefix="$1"
  local -r visibility="$2"

  add_rules_shell "MODULE.bazel"
  write_test "${prefix}/testpkg" "the_test" "${visibility}"
  write_suite "${prefix}/suitepkg" "the_suite" "//${prefix}/testpkg:the_test"
}

function expect_visibility_error() {
  # $1: the label of the suite, $2: the label of the test it can't see. The
  # message spans multiple lines.
  expect_log "target '$2' is not visible from$"
  expect_log "^target '$1'$"
}

function expect_suite_not_reported() {
  # An expanded suite is not a top-level target: it must not appear in the
  # build summary, and no BEP TargetConfigured/TargetCompleted events (nor
  # Aborted events for the latter) may be posted for it.
  local -r suite="$1"
  local -r bep="$2"
  expect_not_log "Target ${suite} up-to-date"
  cat "${bep}" > "$TEST_log"
  expect_not_log "\"targetConfigured\":{\"label\":\"${suite}\""
  expect_not_log "\"targetCompleted\":{\"label\":\"${suite}\""
  expect_not_log "\"aborted\""
}

#### TESTS: visibility violations are caught #################################

function _test_build_catches_private_test_in_suite() {
  local -r pkg="$1"
  write_test_and_suite "$pkg" '["//visibility:private"]'

  bazel build "$2" "//${pkg}/suitepkg:the_suite" &> "$TEST_log" \
      && fail "bazel build should have failed due to a visibility error"
  expect_visibility_error "//${pkg}/suitepkg:the_suite" "//${pkg}/testpkg:the_test"
}

function test_build_catches_private_test_in_suite_skymeld() {
  _test_build_catches_private_test_in_suite "$FUNCNAME" "$SKYMELD"
}

function test_build_catches_private_test_in_suite_noskymeld() {
  _test_build_catches_private_test_in_suite "$FUNCNAME" "$NO_SKYMELD"
}

function _test_test_catches_private_test_in_suite() {
  local -r pkg="$1"
  write_test_and_suite "$pkg" '["//visibility:private"]'

  bazel test "$2" "//${pkg}/suitepkg:the_suite" &> "$TEST_log" \
      && fail "bazel test should have failed due to a visibility error"
  expect_visibility_error "//${pkg}/suitepkg:the_suite" "//${pkg}/testpkg:the_test"
  expect_not_log "//${pkg}/testpkg:the_test\s\+PASSED"
}

function test_test_catches_private_test_in_suite_skymeld() {
  _test_test_catches_private_test_in_suite "$FUNCNAME" "$SKYMELD"
}

function test_test_catches_private_test_in_suite_noskymeld() {
  _test_test_catches_private_test_in_suite "$FUNCNAME" "$NO_SKYMELD"
}

# Baseline: cquery (--noexpand_test_suites) always analyzed the suite.
function test_cquery_catches_private_test_in_suite() {
  local -r pkg="$FUNCNAME"
  write_test_and_suite "$pkg" '["//visibility:private"]'

  bazel cquery "//${pkg}/suitepkg:the_suite" &> "$TEST_log" \
      && fail "bazel cquery should have failed due to a visibility error"
  expect_visibility_error "//${pkg}/suitepkg:the_suite" "//${pkg}/testpkg:the_test"
}

# Baseline: with --noexpand_test_suites the suite is a regular top-level target.
function test_build_noexpand_catches_private_test_in_suite() {
  local -r pkg="$FUNCNAME"
  write_test_and_suite "$pkg" '["//visibility:private"]'

  bazel build --noexpand_test_suites "//${pkg}/suitepkg:the_suite" &> "$TEST_log" \
      && fail "bazel build --noexpand_test_suites should have failed due to a visibility error"
  expect_visibility_error "//${pkg}/suitepkg:the_suite" "//${pkg}/testpkg:the_test"
}

# With --keep_going the violation is still an error, but the other requested
# targets are built and tested. Note that this includes the tests the invalid
# suite expanded to: it's the suite's analysis that failed, not theirs.
function _test_keep_going_still_fails_and_tests_the_rest() {
  local -r pkg="$1"
  write_test_and_suite "$pkg" '["//visibility:private"]'
  write_test "${pkg}/otherpkg" "other_test" '["//visibility:public"]'

  bazel test --keep_going "$2" \
      "//${pkg}/suitepkg:the_suite" "//${pkg}/otherpkg:other_test" &> "$TEST_log" \
      && fail "bazel test --keep_going should have failed due to a visibility error"
  expect_visibility_error "//${pkg}/suitepkg:the_suite" "//${pkg}/testpkg:the_test"
  expect_log "//${pkg}/otherpkg:other_test\s\+PASSED"
  expect_log "//${pkg}/testpkg:the_test\s\+PASSED"
  expect_log "Executed 2 out of 2 tests"
}

function test_keep_going_still_fails_and_tests_the_rest_skymeld() {
  _test_keep_going_still_fails_and_tests_the_rest "$FUNCNAME" "$SKYMELD"
}

function test_keep_going_still_fails_and_tests_the_rest_noskymeld() {
  _test_keep_going_still_fails_and_tests_the_rest "$FUNCNAME" "$NO_SKYMELD"
}

# Nested test_suite: a private test reached only through a nested suite must
# still trigger a visibility error when building the outer suite. Analyzing
# outer_suite pulls inner_suite in as a dependency, whose own `tests`
# references are then visibility-checked.
function test_build_catches_private_test_in_nested_suite() {
  local -r pkg="$FUNCNAME"
  add_rules_shell "MODULE.bazel"
  write_test "${pkg}/testpkg" "the_test" '["//visibility:private"]'
  write_suite "${pkg}/innerpkg" "inner_suite" "//${pkg}/testpkg:the_test"
  write_suite "${pkg}/outerpkg" "outer_suite" "//${pkg}/innerpkg:inner_suite"

  bazel build "//${pkg}/outerpkg:outer_suite" &> "$TEST_log" \
      && fail "bazel build should have failed due to a visibility error"
  # The violation is inner_suite's reference, not outer_suite's.
  expect_visibility_error "//${pkg}/innerpkg:inner_suite" "//${pkg}/testpkg:the_test"
  expect_log "Analysis of target '//${pkg}/outerpkg:outer_suite' failed"
}

#### TESTS: valid suites keep working exactly as before ######################

function _test_visible_test_in_suite_succeeds() {
  local -r pkg="$1"
  local -r bep="${TEST_TMPDIR}/${pkg}.bep.json"
  write_test_and_suite "$pkg" "[\"//${pkg}/suitepkg:__pkg__\"]"

  bazel build "$2" --build_event_json_file="${bep}" \
      "//${pkg}/suitepkg:the_suite" &> "$TEST_log" || fail "bazel build should have succeeded"
  expect_log "Target //${pkg}/testpkg:the_test up-to-date"
  expect_suite_not_reported "//${pkg}/suitepkg:the_suite" "${bep}"

  bazel test "$2" --build_event_json_file="${bep}" \
      "//${pkg}/suitepkg:the_suite" &> "$TEST_log" || fail "bazel test should have succeeded"
  expect_log "//${pkg}/testpkg:the_test\s\+PASSED"
  expect_log "Executed 1 out of 1 test"
  expect_suite_not_reported "//${pkg}/suitepkg:the_suite" "${bep}"
}

function test_visible_test_in_suite_succeeds_skymeld() {
  _test_visible_test_in_suite_succeeds "$FUNCNAME" "$SKYMELD"
}

function test_visible_test_in_suite_succeeds_noskymeld() {
  _test_visible_test_in_suite_succeeds "$FUNCNAME" "$NO_SKYMELD"
}

function test_public_test_in_suite_succeeds() {
  local -r pkg="$FUNCNAME"
  write_test_and_suite "$pkg" '["//visibility:public"]'

  bazel test "//${pkg}/suitepkg:the_suite" &> "$TEST_log" \
      || fail "bazel test should have succeeded"
  expect_log "Executed 1 out of 1 test"
  bazel cquery "//${pkg}/suitepkg:the_suite" &> "$TEST_log" \
      || fail "bazel cquery should have succeeded"
}

# Excluding a test of a suite on the command line must still take effect:
# analyzing the suite (which depends on the excluded test) must neither build
# nor run it.
function _test_target_exclusion_of_test_in_suite() {
  local -r pkg="$1"
  add_rules_shell "MODULE.bazel"
  write_test "${pkg}" "test_in_suite_1" '["//visibility:public"]'
  write_test "${pkg}" "test_in_suite_2" '["//visibility:public"]'
  write_suite "${pkg}" "the_suite" ":test_in_suite_1" ":test_in_suite_2"

  # '--' is required so that '-//pkg:test_in_suite_1' isn't parsed as a flag.
  bazel test "$2" -- "//${pkg}:the_suite" "-//${pkg}:test_in_suite_1" \
      &> "$TEST_log" || fail "bazel test should have succeeded"
  expect_log "//${pkg}:test_in_suite_2\s\+PASSED"
  expect_not_log "//${pkg}:test_in_suite_1"
  expect_log "Executed 1 out of 1 test"
}

function test_target_exclusion_of_test_in_suite_skymeld() {
  _test_target_exclusion_of_test_in_suite "$FUNCNAME" "$SKYMELD"
}

function test_target_exclusion_of_test_in_suite_noskymeld() {
  _test_target_exclusion_of_test_in_suite "$FUNCNAME" "$NO_SKYMELD"
}

# A nested test_suite composition with proper visibility runs the underlying
# test exactly once.
function test_nested_suite_with_visible_tests_succeeds() {
  local -r pkg="$FUNCNAME"
  add_rules_shell "MODULE.bazel"
  write_test "${pkg}/testpkg" "the_test" '["//visibility:public"]'
  write_suite "${pkg}/innerpkg" "inner_suite" "//${pkg}/testpkg:the_test"
  write_suite "${pkg}/outerpkg" "outer_suite" "//${pkg}/innerpkg:inner_suite"

  bazel test "//${pkg}/outerpkg:outer_suite" &> "$TEST_log" \
      || fail "bazel test should have succeeded"
  expect_log "//${pkg}/testpkg:the_test\s\+PASSED"
  expect_log "Executed 1 out of 1 test"
  expect_not_log "Target //${pkg}/outerpkg:outer_suite up-to-date"
  expect_not_log "Target //${pkg}/innerpkg:inner_suite up-to-date"
}

run_suite "Integration tests for test_suite visibility checking"
