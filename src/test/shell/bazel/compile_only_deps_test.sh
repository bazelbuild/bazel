#!/bin/bash
#
# Copyright 2018 The Bazel Authors. All rights reserved.
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
# Tests the proper checking of BUILD and BUILD.bazel files.

set -euo pipefail
# --- begin runfiles.bash initialization ---
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

function create_target_directory {
  mkdir target
}

function create_empty_header {
  name=$1

  touch target/$name".h"
}

function write_header_inclusion {
  name=$1

  cat << EOF
#include "$name.h"
EOF
}

function write_returning_function {
  name=$1
  return_value=$2

  cat << EOF
int $name() {
  return $return_value;
}
EOF
}

function write_build_file {
  binary_name=$1
  compile_hdrs_or_srcs=$2

  compile_h_or_c="h"
  if [[ "${compile_hdrs_or_srcs:0:1}" != "h" ]]; then
    compile_h_or_c="c"
  fi

  cat << EOF
cc_library(
    name = "compile_only_dep",
    $compile_hdrs_or_srcs = ["compile_only_dep.$compile_h_or_c"],
)
EOF

  cat << EOF
cc_library(
    name = "compile_only_dep_includer",
    srcs = ["compile_only_dep_includer.c"],
    compile_only_deps = [":compile_only_dep"],
)
EOF

  cat << EOF
cc_binary(
    name = "$binary_name",
    srcs = ["$binary_name.c"],
    deps = [":compile_only_dep_includer"],
)
EOF
}

# Ensure headers from compile_only_deps dependencies are hidden
function test_header_exclusion {
  create_new_workspace
  create_target_directory

  dep_header_name="compile_only_dep"
  cc_binary_name="compile_hdr_includer"

  create_empty_header "$dep_header_name"
  touch target/compile_only_dep_includer.c
  write_header_inclusion "$dep_header_name" > target/$cc_binary_name.c
  write_returning_function "main" "0" >> target/$cc_binary_name.c
  write_build_file "$cc_binary_name" "hdrs" >> target/BUILD

  bazel build --spawn_strategy=local //target:$cc_binary_name >& $TEST_log && fail "build should fail"
  # XXX THIS WILL NOT FAIL IN THIS WAY UNDER SANDBOXING XXX
  expect_log "undeclared inclusion(s) in rule '//target:$cc_binary_name'"
}

# Ensure symbols from compile_only_deps dependencies are usable
function test_header_inclusion {
  create_new_workspace
  create_target_directory

  dep_header_name="compile_only_dep"
  cc_binary_name="compile_hdr_includer"

  create_empty_header "$dep_header_name"
  write_header_inclusion "$dep_header_name" > target/compile_only_dep_includer.c
  write_returning_function "main" "0" > target/$cc_binary_name.c
  write_build_file "$cc_binary_name" "hdrs" >> target/BUILD

  bazel build //target:$cc_binary_name >& $TEST_log || fail "build should succeed"
}

# Ensure symbols from compile_only_deps dependencies are usable
function test_symbol_inclusion {
  create_new_workspace
  create_target_directory

  dep_source_name="compile_only_dep"
  compile_only_dep_function_name="compile_only_dep_func"
  cc_binary_name="compile_symbol_user"

  write_returning_function "$compile_only_dep_function_name" "0" > target/$dep_source_name.c
  touch target/compile_only_dep_includer.c
  write_returning_function "main" "$compile_only_dep_function_name()" > target/$cc_binary_name.c
  write_build_file "$cc_binary_name" "srcs" >> target/BUILD

  bazel build //target:$cc_binary_name >& $TEST_log || fail "build should succeed"
}

run_suite "compile only deps test"
