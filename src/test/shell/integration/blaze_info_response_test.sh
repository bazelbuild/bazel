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

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

#### TESTS #############################################################

_fine_logging_configured=0

function set_up() {
  if [[ "${_fine_logging_configured}" -eq 1 ]]; then
    return
  fi
  _fine_logging_configured=1

  local output_base
  output_base="$(bazel info output_base)" || fail "get ${PRODUCT_NAME} output_base failed"
  local javalog_props="${output_base}/javalog.properties"
  if [[ -f "${javalog_props}" ]]; then
    # Bazel's generated javalog.properties sets SimpleLogHandler.level=INFO,
    # which filters out FINE logs even when --logging=4 is passed.
    local custom_javalog_props="${TEST_TMPDIR}/custom_javalog.properties"
    sed 's/SimpleLogHandler\.level=INFO/SimpleLogHandler.level=ALL/' \
      "${javalog_props}" > "${custom_javalog_props}"
    if is_windows; then
      custom_javalog_props="$(cygpath -m "${custom_javalog_props}")"
    fi
    add_to_bazelrc "startup --host_jvm_args=-Djava.util.logging.config.file=${custom_javalog_props}"
    bazel shutdown || fail "${PRODUCT_NAME} shutdown failed"
  fi
}

function test_remote_client_info_multiple_keys() {
  bazel info --info_output_type=response_proto --logging=4 \
    release used-heap-size gc-count >"$TEST_log" \
    || fail "${PRODUCT_NAME} info failed"

  SERVER_LOG=$(bazel info server_log) || fail "get ${PRODUCT_NAME} info log failed"

  assert_contains "Blaze info is invoked by a remote client.*print_keys: true" \
    "$SERVER_LOG"
  assert_contains "path_to_replace { type: OUTPUT_BASE value: \".*\" }" \
    "$SERVER_LOG"
  assert_contains "path_to_replace { type: BUILD_WORKSPACE_DIRECTORY value: \".*\" }" \
    "$SERVER_LOG"
  assert_contains "path_to_replace { type: BUILD_WORKING_DIRECTORY value: \".*\" }" \
    "$SERVER_LOG"
  assert_contains "info_item { key: \"release\" value: \"development version" \
    "$SERVER_LOG"
  assert_contains "info_item { key: \"used-heap-size\" value: \"[0-9]*MB" \
    "$SERVER_LOG"
  assert_contains "info_item { key: \"gc-count\" value: \"[0-9]*" \
    "$SERVER_LOG"
}

function test_remote_client_info_multiple_keys_wrong_keys() {
  bazel info --info_output_type=response_proto --logging=4 \
    foo used-heap-size-after-gc bar &>"$TEST_log" \
    && fail "expected ${PRODUCT_NAME} info to fail with unknown keys"

  SERVER_LOG=$(bazel info server_log) || fail "get ${PRODUCT_NAME} info log failed"

  # Although there is only one valid info item in the InfoResponse, the key
  # should still printed to the server log, given multiple items are requested.
  assert_contains "Blaze info is invoked by a remote client.*print_keys: true" \
    "$SERVER_LOG"
  assert_contains "path_to_replace { type: OUTPUT_BASE value: \".*\" }" \
    "$SERVER_LOG"
  assert_contains "path_to_replace { type: BUILD_WORKSPACE_DIRECTORY value: \".*\" }" \
    "$SERVER_LOG"
  assert_contains "path_to_replace { type: BUILD_WORKING_DIRECTORY value: \".*\" }" \
    "$SERVER_LOG"
  assert_contains "info_item { key: \"used-heap-size-after-gc\" value: \"[0-9]*MB" \
    "$SERVER_LOG"

  # Then the error message.
  expect_log "ERROR: unknown key(s): 'foo', 'bar'"
}

run_suite "Bazel info response integration tests."
