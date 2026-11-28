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
# Tests how test-setup.sh finds the test executable in the runfiles tree or,
# without one, in the runfiles manifest. A label cannot contain a backslash or
# a newline, so instead of running a test with Bazel, these tests run
# test-setup.sh on runfiles they create themselves.

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

source "$(rlocation io_bazel/src/test/shell/unittest.bash)" \
  || { echo >&2 "unittest.bash not found!"; exit 1; }
TEST_SETUP="$(rlocation io_bazel/tools/test/test-setup.sh)"

readonly NL=$'\n'

function set_up() {
  WORK="$TEST_TMPDIR/work"
  rm -rf "$WORK"
  # Like the runfiles directory that Bazel creates without a runfiles tree,
  # this one starts with only the manifest and an empty directory for the main
  # repository, so that test-setup.sh has to look tests up in the manifest.
  mkdir -p "$WORK/runfiles/_main"
  MANIFEST="$WORK/runfiles/MANIFEST"
  touch "$MANIFEST"
  # The manifest does not escape spaces in a target, such as those of an output
  # base in a home directory with a space.
  BIN="$WORK/output base/bin"
}

# Usage: add_test <runfiles path> <name>
#
# Creates an executable that prints "ran <name>", at the runfiles path without
# its leading _main/ under $BIN, and lists it in the manifest the way
# SourceManifestAction does: if the runfiles path contains a space or a
# newline, or the target contains a newline, the line starts with a space, and
# backslashes, spaces and newlines in the runfiles path and backslashes and
# newlines in the target are escaped as \b, \s and \n.
function add_test() {
  local path="$1" name="$2"
  local target="$BIN/${path#_main/}"
  mkdir -p "${target%/*}"
  printf '#!/bin/sh\necho "ran %s"\n' "$name" > "$target"
  chmod +x "$target"
  if [[ "$path" == *" "* || "$path" == *"$NL"* || "$target" == *"$NL"* ]]; then
    path="${path//\\/\\b}"
    path="${path// /\\s}"
    path="${path//$NL/\\n}"
    target="${target//\\/\\b}"
    target="${target//$NL/\\n}"
    printf ' %s %s\n' "$path" "$target" >> "$MANIFEST"
  else
    printf '%s %s\n' "$path" "$target" >> "$MANIFEST"
  fi
}

# Usage: run_test_setup <test executable, relative to the main repository>
#
# Runs test-setup.sh on the test executable, with $WORK/runfiles as its
# runfiles directory and nothing else from this test's environment.
function run_test_setup() {
  env -i \
    PATH="$PATH" \
    USER="${USER:-nobody}" \
    TEST_SRCDIR="$WORK/runfiles" \
    RUNFILES_DIR="$WORK/runfiles" \
    TEST_WORKSPACE=_main \
    TEST_TMPDIR="$WORK/tmp" \
    XML_OUTPUT_FILE="$WORK/test.xml" \
    TEST_UNDECLARED_OUTPUTS_DIR="$WORK/outputs" \
    TEST_UNDECLARED_OUTPUTS_MANIFEST="$WORK/outputs_manifest/MANIFEST" \
    TEST_UNDECLARED_OUTPUTS_ANNOTATIONS="$WORK/annotations" \
    TEST_UNDECLARED_OUTPUTS_ANNOTATIONS_DIR="$WORK/annotations_dir" \
    "$TEST_SETUP" "$1" &> "$TEST_log"
}

function test_plain_path() {
  add_test "_main/pkg/test" plain
  run_test_setup "pkg/test" || fail "test-setup.sh failed"
  expect_log "ran plain"
}

function test_path_with_space() {
  add_test "_main/dir with space/test" space
  run_test_setup "dir with space/test" || fail "test-setup.sh failed"
  expect_log "ran space"
}

function test_path_with_newline() {
  add_test "_main/dir${NL}with${NL}newline/test" newline
  run_test_setup "dir${NL}with${NL}newline/test" || fail "test-setup.sh failed"
  expect_log "ran newline"
}

function test_path_with_backslash() {
  # Without a space or a newline, the path is not escaped in the manifest.
  add_test '_main/dir\with\backslash/test' backslash
  run_test_setup 'dir\with\backslash/test' || fail "test-setup.sh failed"
  expect_log "ran backslash"
}

function test_path_with_backslash_and_space() {
  add_test '_main/dir\with backslash/test' backslash_and_space
  run_test_setup 'dir\with backslash/test' || fail "test-setup.sh failed"
  expect_log "ran backslash_and_space"
}

function test_path_with_backslash_and_newline() {
  add_test "_main/dir\\with${NL}newline/test" backslash_and_newline
  run_test_setup "dir\\with${NL}newline/test" || fail "test-setup.sh failed"
  expect_log "ran backslash_and_newline"
}

function test_path_with_escape_sequences() {
  # Nothing is decoded in an unescaped line, and each backslash is escaped in
  # an escaped one, so none of these turn into a newline, a space or a lone
  # backslash.
  add_test '_main/dir\n\s\b/test' unescaped
  add_test '_main/dir\n\s\b with space/test' escaped
  run_test_setup 'dir\n\s\b/test' || fail "test-setup.sh failed"
  expect_log "ran unescaped"
  run_test_setup 'dir\n\s\b with space/test' || fail "test-setup.sh failed"
  expect_log "ran escaped"
}

function test_path_with_regex_metacharacters() {
  # Listed first, so that they are found if a dot matches any character.
  add_test '_main/dirXwith[ab]*^$/test' wrong_plain
  add_test '_main/dirXwith[ab]*^$ space/test' wrong_escaped
  add_test '_main/dir.with[ab]*^$/test' right_plain
  add_test '_main/dir.with[ab]*^$ space/test' right_escaped
  run_test_setup 'dir.with[ab]*^$/test' || fail "test-setup.sh failed"
  expect_log "ran right_plain"
  run_test_setup 'dir.with[ab]*^$ space/test' || fail "test-setup.sh failed"
  expect_log "ran right_escaped"
}

function test_path_with_space_in_runfiles_tree() {
  # A test in the runfiles tree runs from there, not from the manifest's
  # target.
  add_test "_main/dir with space/test" manifest
  local tree="$WORK/runfiles/_main/dir with space/test"
  mkdir -p "${tree%/*}"
  printf '#!/bin/sh\necho "ran tree"\n' > "$tree"
  chmod +x "$tree"
  run_test_setup "dir with space/test" || fail "test-setup.sh failed"
  expect_log "ran tree"
}

run_suite "test-setup.sh tests"
