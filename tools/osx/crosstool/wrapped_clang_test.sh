#!/bin/bash
# -*- coding: utf-8 -*-

# Copyright 2019 The Bazel Authors. All rights reserved.
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

# Unit tests for wrapped_clang.

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


# Load test environment
source "$(rlocation "io_bazel/src/test/shell/unittest.bash")" \
  || { echo "unittest.bash not found!" >&2; exit 1; }
WRAPPED_CLANG=$(rlocation "io_bazel/tools/osx/crosstool/wrapped_clang")


# This env var tells wrapped_clang to log its command instead of running.
export __WRAPPED_CLANG_LOG_ONLY=1


# Test that add_ast_path is remapped properly.
function test_add_ast_path_remapping() {
  env DEVELOPER_DIR=dummy SDKROOT=a \
      "${WRAPPED_CLANG}" "-Wl,-add_ast_path,foo" >$TEST_log || fail "wrapped_clang failed";
  expect_log "-Wl,-add_ast_path,${PWD}/foo" "Expected add_ast_path to be remapped."
}

function test_disable_add_ast_path_remapping() {
  env RELATIVE_AST_PATH=isset DEVELOPER_DIR=dummy SDKROOT=a \
      "${WRAPPED_CLANG}" "-Wl,-add_ast_path,relative/foo" >$TEST_log || fail "wrapped_clang failed";
  expect_log "-Wl,-add_ast_path,relative/foo" "Expected add_ast_path to not be remapped."
}

# Test that __BAZEL_XCODE_DEVELOPER_DIR__ is remapped properly.
function test_developer_dir_remapping() {
  env DEVELOPER_DIR=mydir SDKROOT=a \
      "${WRAPPED_CLANG}" "developer_dir=__BAZEL_XCODE_DEVELOPER_DIR__" \
      >$TEST_log || fail "wrapped_clang failed";
  expect_log "developer_dir=mydir" "Expected developer dir to be remapped."
}

# Test that __BAZEL_XCODE_SDKROOT__ is remapped properly.
function test_sdkroot_remapping() {
  env DEVELOPER_DIR=dummy SDKROOT=mysdkroot \
      "${WRAPPED_CLANG}" "sdkroot=__BAZEL_XCODE_SDKROOT__" \
      >$TEST_log || fail "wrapped_clang failed";
  expect_log "sdkroot=mysdkroot" "Expected sdkroot to be remapped."
}

function test_params_expansion() {
  params=$(mktemp)
  {
    echo "first"
    echo "-rpath"
    echo "@loader_path"
    echo "sdkroot=__BAZEL_XCODE_SDKROOT__"
    echo "developer_dir=__BAZEL_XCODE_DEVELOPER_DIR__"
  } > "$params"

  env DEVELOPER_DIR=dummy SDKROOT=mysdkroot \
      "${WRAPPED_CLANG}" "@$params" \
      >"$TEST_log" || fail "wrapped_clang failed";
  expect_log "/usr/bin/xcrun clang first -rpath @loader_path sdkroot=mysdkroot developer_dir=dummy"
}

run_suite "Wrapped clang tests"
