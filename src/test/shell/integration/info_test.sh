#!/usr/bin/env bash
#
# Copyright 2020 The Bazel Authors. All rights reserved.
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
# An end-to-end test that Bazel info command reasonable output.

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


#### TESTS #############################################################

function tear_down() {
  if [[ -n "${outside_dir:-}" ]]; then
    rm -rf "${outside_dir}"
    unset outside_dir
  fi
}

function test_info() {
  bazel info >$TEST_log \
    || fail "${PRODUCT_NAME} info failed"

  # Test some arbitrary keys.
  expect_log 'max-heap-size: [0-9]*MB'
  expect_log 'server_pid: [0-9]*'
  expect_log 'command_log: .*/command\.log'
  expect_log 'release: development version'

  # Make sure that hidden keys are not shown.
  expect_not_log 'used-heap-size-after-gc'
  expect_not_log 'starlark-semantics'
}

function test_server_pid() {
  bazel info server_pid >$TEST_log \
    || fail "${PRODUCT_NAME} info failed"
  expect_log '[0-9]*'
}

function test_external_directory() {
  bazel info external_directory >$TEST_log \
    || fail "${PRODUCT_NAME} info failed"
  local -r output_base="$(bazel info output_base 2>/dev/null)"
  expect_log "^${output_base}/external\$"
}

function test_used_heap_size_after_gc() {
  bazel info used-heap-size-after-gc >$TEST_log \
    || fail "${PRODUCT_NAME} info failed"
  expect_log '[0-9]*MB'
}

function test_starlark_semantics() {
  bazel info starlark-semantics >$TEST_log \
    || fail "${PRODUCT_NAME} info failed"
  expect_log 'StarlarkSemantics{.*}'
}

function test_multiple_keys() {
  bazel info release used-heap-size gc-count >$TEST_log \
    || fail "${PRODUCT_NAME} info failed"
  expect_log 'release: development version'
  expect_log 'used-heap-size: [0-9]*MB'
  expect_log 'gc-count: [0-9]*'
}

function test_multiple_keys_wrong_keys() {
  bazel info command_log foo used-heap-size-after-gc bar gc-count foo &>$TEST_log \
    && fail "expected ${PRODUCT_NAME} info to fail with unknown keys"

  # First test the valid keys.
  expect_log 'command_log: .*/command\.log'
  expect_log 'used-heap-size-after-gc: [0-9]*MB'
  expect_log 'gc-count: [0-9]*'

  # Then the error message.
  expect_log "ERROR: unknown key(s): 'foo', 'bar'"
}

# Regression test for https://github.com/bazelbuild/bazel/issues/24671
function test_invalid_flag_error() {
  # This type of loading error only happens with external dependencies.
  if [[ "$PRODUCT_NAME" != "bazel" ]]; then
    return 0
  fi
  bazel info --registry=foobarbaz &>$TEST_log \
    && fail "expected ${PRODUCT_NAME} to fail with an invalid registry"
  expect_not_log "crashed due to an internal error"
  expect_log "Invalid registry URL: foobarbaz"
}

# Regression test for https://github.com/bazelbuild/bazel/issues/14001
function test_info_outside_workspace() {
  outside_dir="$(mktemp -d /tmp/outside_workspace.XXXXXX)"

  (cd "${outside_dir}" && bazel info) &> "$TEST_log" \
    || fail "Expected info to work outside a workspace"

  # Workspace-independent keys should be present.
  expect_log 'install_base: '
  expect_log 'output_base: '
  expect_log 'release: '
  expect_log 'server_pid: [0-9]*'

  # Workspace-dependent keys should NOT be present.
  expect_not_log 'workspace:'
  expect_not_log "${PRODUCT_NAME}-bin:"
  expect_not_log "${PRODUCT_NAME}-genfiles:"
  expect_not_log "${PRODUCT_NAME}-testlogs:"
  expect_not_log 'execution_root:'
  expect_not_log 'output_path:'
  expect_not_log 'package_path:'
}

function test_info_install_base_outside_workspace() {
  outside_dir="$(mktemp -d /tmp/outside_workspace.XXXXXX)"

  (cd "${outside_dir}" && bazel info install_base) > "$TEST_log" \
    || fail "Expected info install_base to work outside a workspace"

  local install_base
  install_base="$(cat "$TEST_log")"
  [[ -d "${install_base}" ]] || fail "install_base '${install_base}' is not a directory"
}

function test_info_workspace_key_outside_workspace() {
  outside_dir="$(mktemp -d /tmp/outside_workspace.XXXXXX)"

  (cd "${outside_dir}" && bazel info workspace) &> "$TEST_log" \
    && fail "Expected bazel info workspace outside a workspace to fail"
  assert_equals 2 "$?"
  expect_log "ERROR: key(s) only supported from within a workspace (below a directory having a MODULE\.bazel file): 'workspace'"
}

function test_info_product_bin_outside_workspace() {
  outside_dir="$(mktemp -d /tmp/outside_workspace.XXXXXX)"

  (cd "${outside_dir}" && bazel info "${PRODUCT_NAME}-bin") &> "$TEST_log" \
    && fail "Expected bazel info ${PRODUCT_NAME}-bin outside a workspace to fail"
  assert_equals 2 "$?"
  expect_log "ERROR: key(s) only supported from within a workspace (below a directory having a MODULE\.bazel file): '${PRODUCT_NAME}-bin'"
}

function test_info_mixed_keys_outside_workspace() {
  outside_dir="$(mktemp -d /tmp/outside_workspace.XXXXXX)"

  (cd "${outside_dir}" && bazel info release "${PRODUCT_NAME}-bin" package_path) &> "$TEST_log" \
    && fail "Expected bazel info with workspace keys outside a workspace to fail"
  assert_equals 2 "$?"
  expect_log "release: "
  expect_log "ERROR: key(s) only supported from within a workspace (below a directory having a MODULE\.bazel file): '${PRODUCT_NAME}-bin', 'package_path'"
}

function test_info_show_make_env_outside_workspace() {
  outside_dir="$(mktemp -d /tmp/outside_workspace.XXXXXX)"

  (cd "${outside_dir}" && bazel info --show_make_env) &> "$TEST_log" \
    && fail "Expected bazel info --show_make_env outside a workspace to fail"
  assert_equals 2 "$?"
  expect_log "ERROR: option '--show_make_env' is only supported from within a workspace (below a directory having a MODULE\.bazel file)"
}

run_suite "Integration tests for ${PRODUCT_NAME} info."
