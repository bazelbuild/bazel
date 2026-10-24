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

# --- begin runfiles.bash initialization v3 ---
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

source "$(rlocation io_bazel/src/test/shell/unittest.bash)" \
  || (echo "unittest.bash not found!" && exit 1)

script="$(rlocation io_bazel/combine_distfiles_to_tar.sh)"

function create_zip() {
  local archive="$1"
  local input_dir="$2"
  mkdir -p "$input_dir"
  printf 'contents\n' > "$input_dir/file.txt"
  (cd "$input_dir" && zip -q "$archive" file.txt)
}

function assert_archive_contents() {
  local archive="$1"
  assert_equals './file.txt' "$(tar -tf "$archive")"
}

function test_relative_archive_path_with_spaces() {
  local test_root="$TEST_TMPDIR/relative paths"
  mkdir -p "$test_root"
  create_zip "$test_root/input archive.zip" "$test_root/input"

  (cd "$test_root" && "$script" "output archive.tar" "input archive.zip")

  assert_archive_contents "$test_root/output archive.tar"
}

function test_absolute_paths_and_tmpdir_with_spaces() {
  local test_root="$TEST_TMPDIR/absolute paths"
  local temp_dir="$test_root/temp dir"
  local input_archive="$test_root/input archive.zip"
  local output_archive="$test_root/output archive.tar"
  mkdir -p "$temp_dir"
  create_zip "$input_archive" "$test_root/input"

  TMPDIR="$temp_dir" "$script" "$output_archive" "$input_archive"

  assert_archive_contents "$output_archive"
}

run_suite "combine_distfiles_to_tar.sh tests"
