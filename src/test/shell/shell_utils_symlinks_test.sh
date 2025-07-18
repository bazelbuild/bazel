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

function test_resolve_bad_links() {
  local -r dir="${FUNCNAME[0]}"

  mkdir -p "$dir" || fail "mkdir -p $dir"

  # absolute, non-existent path
  assert_fails_to resolve_links "$(pwd)/${dir}/non-existent"

  # relative, non-existent path
  assert_fails_to resolve_links "${dir}/non-existent"

  # symlink with absolute non-existent target
  ln -s "/non-existent" "${dir}/bad-absolute.sym"
  assert_fails_to resolve_links "$(pwd)/${dir}/bad-absolute.sym"
  assert_fails_to resolve_links "${dir}/bad-absolute.sym"

  # symlink with relative non-existent target
  ln -s "non-existent" "${dir}/bad-relative.sym"
  assert_fails_to resolve_links "$(pwd)/${dir}/bad-relative.sym"
  assert_fails_to resolve_links "${dir}/bad-relative.sym"

  # circular symlink
  ln -s "circular.sym" "${dir}/circular.sym"
  assert_fails_to resolve_links "$(pwd)/${dir}/circular.sym"
  assert_fails_to resolve_links "${dir}/circular.sym"
}


function test_resolve_symlinks() {
  local -r dir="${FUNCNAME[0]}"

  mkdir -p "${dir}/a/b" || fail "mkdir -p ${dir}/a/b"
  echo hello > "${dir}/hello.txt"

  ln -s "." "${dir}/self"
  ln -s "../hello.txt" "${dir}/a/sym"
  ln -s "../sym" "${dir}/a/b/sym"
  ln -s ".././sym" "${dir}/a/b/sym-not-normalized"

  assert_equals "${dir}/." "$(resolve_links "${dir}/self")"
  assert_equals "${dir}/a/../hello.txt" "$(resolve_links "${dir}/a/sym")"
  assert_equals "${dir}/a/b/../../hello.txt" "$(resolve_links "${dir}/a/b/sym")"
  assert_equals \
      "${dir}/a/b/.././../hello.txt" \
      "$(resolve_links "${dir}/a/b/sym-not-normalized")"

  cd "$dir"
  assert_equals "./." "$(resolve_links "self")"
  assert_equals "./." "$(resolve_links "./self")"
  assert_equals "a/../hello.txt" "$(resolve_links "a/sym")"
  assert_equals "a/b/../../hello.txt" "$(resolve_links "a/b/sym")"
  assert_equals \
      "a/b/.././../hello.txt" \
      "$(resolve_links "a/b/sym-not-normalized")"

  cd a
  assert_equals "../." "$(resolve_links "../self")"
  assert_equals "./../hello.txt" "$(resolve_links "sym")"
  assert_equals "./../hello.txt" "$(resolve_links "./sym")"
  assert_equals "b/../../hello.txt" "$(resolve_links "b/sym")"
  assert_equals \
      "b/.././../hello.txt" \
      "$(resolve_links "b/sym-not-normalized")"

  cd b
  assert_equals "../../." "$(resolve_links "../../self")"
  assert_equals "../../hello.txt" "$(resolve_links "../sym")"
  assert_equals "./../../hello.txt" "$(resolve_links "sym")"
  assert_equals "./../../hello.txt" "$(resolve_links "./sym")"
  assert_equals \
      "./.././../hello.txt" \
      "$(resolve_links "sym-not-normalized")"
}

function test_get_realpath() {
  local -r dir="${FUNCNAME[0]}"

  mkdir -p "${dir}/a/b" || fail "mkdir -p ${dir}/a/b"
  echo hello > "${dir}/hello.txt"

  ln -s "." "${dir}/self"
  ln -s "../hello.txt" "${dir}/a/sym"
  ln -s "../sym" "${dir}/a/b/sym"
  ln -s ".././sym" "${dir}/a/b/sym-not-normalized"

  assert_equals "$(pwd)/${dir}" "$(get_real_path "${dir}/self")"
  assert_equals "$(pwd)/${dir}/hello.txt" "$(get_real_path "${dir}/a/sym")"
  assert_equals "$(pwd)/${dir}/hello.txt" "$(get_real_path "${dir}/a/b/sym")"
  assert_equals \
      "$(pwd)/${dir}/hello.txt" \
      "$(get_real_path "${dir}/a/b/sym-not-normalized")"

  cd "$dir"
  local -r abs_dir=$(pwd)
  assert_equals "${abs_dir}" "$(get_real_path "self")"
  assert_equals "${abs_dir}" "$(get_real_path "./self")"
  assert_equals "${abs_dir}/hello.txt" "$(get_real_path "a/sym")"
  assert_equals "${abs_dir}/hello.txt" "$(get_real_path "a/b/sym")"
  assert_equals \
      "${abs_dir}/hello.txt" "$(get_real_path "a/b/sym-not-normalized")"

  cd a
  assert_equals "${abs_dir}" "$(get_real_path "../self")"
  assert_equals "${abs_dir}/hello.txt" "$(get_real_path "sym")"
  assert_equals "${abs_dir}/hello.txt" "$(get_real_path "./sym")"
  assert_equals "${abs_dir}/hello.txt" "$(get_real_path "b/sym")"
  assert_equals "${abs_dir}/hello.txt" "$(get_real_path "b/sym-not-normalized")"

  cd b
  assert_equals "${abs_dir}" "$(get_real_path "../../self")"
  assert_equals "${abs_dir}/hello.txt" "$(get_real_path "../sym")"
  assert_equals "${abs_dir}/hello.txt" "$(get_real_path "sym")"
  assert_equals "${abs_dir}/hello.txt" "$(get_real_path "./sym")"
  assert_equals "${abs_dir}/hello.txt" "$(get_real_path "sym-not-normalized")"

  assert_fails_to get_real_path "non-existent"
  ln -s self self
  assert_fails_to get_real_path "self"
}

run_suite "Tests for Bash utilities"
