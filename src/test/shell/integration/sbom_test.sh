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

function write_app() {
  local pkg="$1"
  mkdir -p "$pkg" || fail "mkdir -p $pkg"
  cat > "$pkg/BUILD" <<'EOF'
genrule(
    name = "app",
    outs = ["app.txt"],
    cmd = "echo app > $@",
)
EOF
}

function assert_sbom_contains() {
  local needle="$1"
  local file="$2"
  grep -aq -- "$needle" "$file" || fail "expected '$needle' in $file"
}

function test_sbom_output_written() {
  local pkg="${FUNCNAME[0]}"
  write_app "$pkg"
  local out="$TEST_TMPDIR/sbom.bin"

  bazel build --sbom_output="$out" "//$pkg:app" &> "$TEST_log" || fail "Expected success"

  [[ -s "$out" ]] || fail "SBOM not written to $out"
  assert_sbom_contains "//$pkg:app" "$out"
  assert_sbom_contains "bazel" "$out"
}

function test_sbom_off_by_default() {
  local pkg="${FUNCNAME[0]}"
  write_app "$pkg"
  local out="$TEST_TMPDIR/sbom.bin"

  bazel build "//$pkg:app" &> "$TEST_log" || fail "Expected success"

  [[ ! -e "$out" ]] || fail "SBOM written without the flag"
}

function test_sbom_deterministic() {
  local pkg="${FUNCNAME[0]}"
  write_app "$pkg"
  local first="$TEST_TMPDIR/first.bin"
  local second="$TEST_TMPDIR/second.bin"

  SOURCE_DATE_EPOCH=1700000000 bazel build --sbom_output="$first" "//$pkg:app" \
    &> "$TEST_log" || fail "Expected success"
  SOURCE_DATE_EPOCH=1700000000 bazel build --sbom_output="$second" "//$pkg:app" \
    &> "$TEST_log" || fail "Expected success"

  cmp -s "$first" "$second" || fail "SBOM output is not deterministic"
}

run_suite "sbom tests"
