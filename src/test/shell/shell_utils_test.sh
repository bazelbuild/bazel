#!/usr/bin/env bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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
# This test exercises Bash utility implementations.

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

source "$(rlocation "io_bazel/src/test/shell/unittest.bash")" \
  || { echo "Could not source unittest.bash" >&2; exit 1; }
source "$(rlocation "io_bazel/src/test/shell/shell_utils.sh")" \
  || { echo "Could not source shell_utils.sh" >&2; exit 1; }

cd "$TEST_TMPDIR"

function assert_fails_to() {
  local -r method="$1"
  local -r arg="${2:-}"
  if [[ -n "$arg" ]]; then
    "$method" "$arg" && fail "'$method' should have failed for '$arg'"
  else
    "$method" && fail "'$method' should have failed for empty argument"
  fi
  true  # reset the exit status otherwise the test would be considered failed
}

function test_resolve_non_links() {
  local -r dir="${FUNCNAME[0]}"

  mkdir -p "$dir" || fail "mkdir -p $dir"
  echo hello > "${dir}/hello.txt"

  # absolute path to directory
  assert_equals "$(pwd)" "$(resolve_links "$(pwd)")"

  # relative path to directory
  assert_equals "${dir}" "$(resolve_links "$dir")"

  # relative path to file
  assert_equals "${dir}/hello.txt" "$(resolve_links "${dir}/hello.txt")"

  # absolute path to file
  assert_equals \
      "$(pwd)/${dir}/hello.txt" "$(resolve_links "$(pwd)/${dir}/hello.txt")"
}

function test_normalize_path() {
  assert_equals "." "$(normalize_path "")"
  assert_equals "." "$(normalize_path ".")"
  assert_equals "." "$(normalize_path "./.")"
  assert_equals ".." "$(normalize_path "..")"
  assert_equals ".." "$(normalize_path "./..")"
  assert_equals ".." "$(normalize_path "../.")"
  assert_equals "../.." "$(normalize_path "../././..")"

  assert_equals "blah" "$(normalize_path "blah")"
  assert_equals "blah" "$(normalize_path "blah/.")"
  assert_equals "blah" "$(normalize_path "blah/./hello/..")"
  assert_equals "blah" "$(normalize_path "blah/./hello/../")"
  assert_equals \
      "blah/hello" "$(normalize_path "blah/./hello/../a/b/.././.././hello")"
  assert_equals "." "$(normalize_path "blah/./hello/../a/b/.././.././..")"
  assert_equals ".." "$(normalize_path "blah/.././..")"
  assert_equals "../.." "$(normalize_path "blah/.././../..")"

  assert_equals "/" "$(normalize_path "/")"
  assert_equals "/" "$(normalize_path "/.")"
  assert_equals "/" "$(normalize_path "/./.")"
  assert_equals "/blah" "$(normalize_path "/blah")"
  assert_equals "/blah" "$(normalize_path "/blah/.")"
  assert_equals "/blah" "$(normalize_path "/blah/./hello/..")"
  assert_equals "/blah" "$(normalize_path "/blah/./hello/../")"
  assert_equals "/blah" "$(normalize_path "/blah/./hello/../a/b/.././../.")"
}

function test_md5_sum() {
  local -r dir="${FUNCNAME[0]}"
  mkdir "$dir" || fail "mkdir $dir"

  echo hello > "${dir}/a.txt"
  echo world > "${dir}/b.txt"

  assert_fails_to md5_file
  assert_fails_to md5_file "non-existent"

  assert_equals "b1946ac92492d2347c6235b4d2611184" "$(md5_file "${dir}/a.txt")"
  assert_equals "591785b794601e212b260e25925636fd" "$(md5_file "${dir}/b.txt")"

  local sums="$(echo -e \
      "b1946ac92492d2347c6235b4d2611184\n591785b794601e212b260e25925636fd")"
  assert_equals "$sums" "$(md5_file "${dir}/a.txt" "${dir}/b.txt")"
}

run_suite "Tests for Bash utilities"
