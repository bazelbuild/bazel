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

function set_up() {
  add_rules_java MODULE.bazel
  add_rules_cc MODULE.bazel
}

function write_cc_graph() {
  local pkg="$1"
  mkdir -p "$pkg"
  cat > "$pkg/BUILD" <<EOF
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

cc_library(
    name = "lib",
    srcs = ["lib.cc"],
)

cc_binary(
    name = "app",
    srcs = ["main.cc"],
    deps = [":lib"],
)
EOF
  cat > "$pkg/lib.cc" <<'EOF'
int lib_fn() { return 42; }
EOF
  cat > "$pkg/main.cc" <<'EOF'
int lib_fn();
int main() { return lib_fn(); }
EOF
}

function assert_compdb_contains_sources() {
  local workspace="$1"
  local pkg="$2"
  local main_file="$3"
  local lib_file="$4"

  python3 - "$workspace" "$main_file" "$lib_file" <<'PY'
import json
import sys

workspace, main_file, lib_file = sys.argv[1:4]
with open("compile_commands.json") as f:
    entries = json.load(f)

by_file = {entry["file"]: entry for entry in entries}
for path in (main_file, lib_file):
    if path not in by_file:
        raise SystemExit(f"missing compile_commands entry for {path}")

for path in (main_file, lib_file):
    entry = by_file[path]
    if entry["directory"] != workspace:
        raise SystemExit(
            f"expected directory {workspace} for {path}, got {entry['directory']}"
        )
    arguments = entry.get("arguments")
    if not arguments:
        raise SystemExit(f"missing arguments for {path}")
    if "-c" not in arguments:
        raise SystemExit(f"missing -c in arguments for {path}")
PY
}

function test_basic_compdb() {
  local pkg="${FUNCNAME[0]}"
  write_cc_graph "$pkg"

  bazel compdb "//$pkg:app" >> "$TEST_log" 2>&1 \
    || fail "Expected compdb to succeed"

  assert_exists compile_commands.json
  assert_compdb_contains_sources "$(pwd)" "$pkg" "$pkg/main.cc" "$pkg/lib.cc"
}

function test_compdb_custom_output() {
  local pkg="${FUNCNAME[0]}"
  write_cc_graph "$pkg"

  bazel compdb --compdb_output="$pkg/out.json" "//$pkg:app" >> "$TEST_log" 2>&1 \
    || fail "Expected compdb to succeed"

  assert_exists "$pkg/out.json"
  assert_not_exists compile_commands.json
  mv "$pkg/out.json" compile_commands.json
  assert_compdb_contains_sources "$(pwd)" "$pkg" "$pkg/main.cc" "$pkg/lib.cc"
}

function test_compdb_stdout() {
  local pkg="${FUNCNAME[0]}"
  write_cc_graph "$pkg"

  bazel compdb --compdb_output=- "//$pkg:app" > "$pkg/compdb.json" 2>> "$TEST_log" \
    || fail "Expected compdb to succeed"

  assert_exists "$pkg/compdb.json"
  mv "$pkg/compdb.json" compile_commands.json
  assert_compdb_contains_sources "$(pwd)" "$pkg" "$pkg/main.cc" "$pkg/lib.cc"
}

run_suite "${PRODUCT_NAME} compdb tests"
