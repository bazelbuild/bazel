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

# Regression test for https://github.com/bazelbuild/bazel/issues/29176 and
# https://github.com/bazelbuild/bazel/issues/22360.
function write_incremental_info_repro() {
  mkdir -p foo
  cat > foo/BUILD <<'EOF'
genrule(
    name = "foo",
    outs = ["out"],
    cmd = "touch $@",
)
EOF
}

function write_flag_alias_repro() {
  mkdir -p flags
  cat > flags/rules.bzl <<'EOF'
starlark_string_flag = rule(
    implementation = lambda ctx: [],
    build_setting = config.string(flag = True),
)
EOF
  cat > flags/BUILD <<'EOF'
load(":rules.bzl", "starlark_string_flag")

starlark_string_flag(
    name = "module_flag",
    build_setting_default = "module_default",
)

starlark_string_flag(
    name = "user_flag",
    build_setting_default = "user_default",
)
EOF
  cat > MODULE.bazel <<'EOF'
module(name = "test")

flag_alias(name = "module_alias", starlark_flag = "//flags:module_flag")
EOF
}

function enable_disk_cache_in_test_bazelrc() {
  echo "common --disk_cache=${TEST_TMPDIR}/disk-cache" >> "${TEST_TMPDIR}/bazelrc"
}

function run_nobuild_with_bep() {
  local bep_file="$1"
  local log_file="$2"
  shift 2
  bazel build --nobuild --build_event_text_file="${bep_file}" "$@" >"${log_file}" 2>&1 \
    || fail "${PRODUCT_NAME} build failed"
}

function assert_targets_configured_metric_absent() {
  local bep_file="$1"
  if grep -q "targets_configured:" "${bep_file}"; then
    cat "${bep_file}" >&2
    fail "expected targets_configured metric to be absent in ${bep_file}"
  fi
}

function assert_targets_configured_metric_present() {
  local bep_file="$1"
  if ! grep -Eq "targets_configured: [1-9][0-9]*" "${bep_file}"; then
    cat "${bep_file}" >&2
    fail "expected a non-zero targets_configured metric in ${bep_file}"
  fi
}

function assert_analyzed_target_summary() {
  local log_file="$1"
  local expected_targets_configured="$2"
  grep -Eq \
    "Analyzed target //foo:foo \\(0 packages loaded, ${expected_targets_configured} target[s]? configured\\)\\." \
    "${log_file}" \
    || fail "expected analyzed target summary with ${expected_targets_configured} target(s) configured in ${log_file}"
}

function test_info_bazel_bin_keeps_incremental_analysis_warm() {
  write_incremental_info_repro
  enable_disk_cache_in_test_bazelrc

  bazel build --nobuild //foo:foo >"${TEST_log}" 2>&1 \
    || fail "${PRODUCT_NAME} initial build failed"
  run_nobuild_with_bep warm.txt warm.log //foo:foo
  assert_targets_configured_metric_absent warm.txt
  assert_analyzed_target_summary warm.log 0

  bazel info bazel-bin >"${TEST_log}" 2>&1 \
    || fail "${PRODUCT_NAME} info bazel-bin failed"
  run_nobuild_with_bep post.txt post.log //foo:foo
  assert_targets_configured_metric_absent post.txt
  assert_analyzed_target_summary post.log 0
}

function test_info_output_base_keeps_incremental_analysis_warm() {
  write_incremental_info_repro
  enable_disk_cache_in_test_bazelrc

  bazel build --nobuild //foo:foo >"${TEST_log}" 2>&1 \
    || fail "${PRODUCT_NAME} initial build failed"
  run_nobuild_with_bep warm.txt warm.log //foo:foo
  assert_targets_configured_metric_absent warm.txt
  assert_analyzed_target_summary warm.log 0

  bazel info output_base >"${TEST_log}" 2>&1 \
    || fail "${PRODUCT_NAME} info output_base failed"
  run_nobuild_with_bep post.txt post.log //foo:foo
  assert_targets_configured_metric_absent post.txt
  assert_analyzed_target_summary post.log 0
}

function test_info_bazel_bin_keeps_incremental_analysis_warm_with_module_and_user_flag_aliases() {
  write_incremental_info_repro
  write_flag_alias_repro
  enable_disk_cache_in_test_bazelrc
  echo "common --flag_alias=user_alias=//flags:user_flag" >> "${TEST_TMPDIR}/bazelrc"

  bazel build --nobuild //foo:foo >"${TEST_log}" 2>&1 \
    || fail "${PRODUCT_NAME} initial build failed"
  run_nobuild_with_bep warm.txt warm.log //foo:foo
  assert_targets_configured_metric_absent warm.txt
  assert_analyzed_target_summary warm.log 0

  bazel info bazel-bin >"${TEST_log}" 2>&1 \
    || fail "${PRODUCT_NAME} info bazel-bin failed"
  run_nobuild_with_bep post.txt post.log //foo:foo
  assert_targets_configured_metric_absent post.txt
  assert_analyzed_target_summary post.log 0
}

run_suite "Integration tests for ${PRODUCT_NAME} info."
