#!/usr/bin/env bash
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
  || { echo "Could not source unittest.bash" >&2; exit 1; }

readonly SCRIPT="$(rlocation io_bazel/src/combine_derived_java_srcs.sh)"

function create_fixture() {
  local -r test_dir="$1"
  local -r javabase="$test_dir/fake jdk"

  mkdir -p "$javabase/bin" "$test_dir/bin" "$test_dir/work"

  cat >"$javabase/bin/jar" <<'EOF'
#!/bin/sh
set -eu
[ "$#" -eq 2 ]
[ "$1" = "xf" ]
[ -f "$2" ]
printf '%s\n' "$2" >>"${JAR_LOG}"
EOF
  chmod +x "$javabase/bin/jar"

  cat >"$test_dir/bin/zip" <<'EOF'
#!/bin/sh
set -eu
output=
for arg in "$@"; do
  output="$arg"
done
cat >"${ZIP_INPUT}"
: >"${output}"
EOF
  chmod +x "$test_dir/bin/zip"

  touch "$test_dir/work/first source.jar" "$test_dir/work/second.jar"
}

function assert_inputs() {
  local -r test_dir="$1"
  assert_equals \
    "$(printf '%s\n' "$test_dir/work/first source.jar" "$test_dir/work/second.jar")" \
    "$(cat "$test_dir/jar.log")"
}

function test_relative_paths_with_spaces() {
  local -r test_dir="$TEST_TMPDIR/${FUNCNAME[0]}"
  local -r javabase="$test_dir/fake jdk"
  export JAR_LOG="$test_dir/jar.log"
  export ZIP_INPUT="$test_dir/zip.input"
  create_fixture "$test_dir"

  (cd "$test_dir/work" && \
    PATH="$test_dir/bin:$PATH" "$SCRIPT" \
      "$javabase" "combined sources.zip" "first source.jar" "second.jar")

  assert_inputs "$test_dir"
  [[ -f "$test_dir/work/combined sources.zip" ]] \
    || fail "combined sources.zip was not created"
}

function test_absolute_paths_and_tmpdir_with_spaces() {
  local -r test_dir="$TEST_TMPDIR/${FUNCNAME[0]}"
  local -r javabase="$test_dir/fake jdk"
  local -r temp_dir="$test_dir/temp dir"
  local -r output="$test_dir/combined sources.zip"
  export JAR_LOG="$test_dir/jar.log"
  export ZIP_INPUT="$test_dir/zip.input"
  create_fixture "$test_dir"
  mkdir -p "$temp_dir"

  PATH="$test_dir/bin:$PATH" TMPDIR="$temp_dir" "$SCRIPT" \
    "$javabase" "$output" \
    "$test_dir/work/first source.jar" "$test_dir/work/second.jar"

  assert_inputs "$test_dir"
  [[ -f "$output" ]] || fail "$output was not created"
}

run_suite "combine_derived_java_srcs.sh tests"
