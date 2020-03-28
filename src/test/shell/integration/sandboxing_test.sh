#!/bin/bash
#
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
#
# Test sandboxing spawn strategy
#

set -euo pipefail

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function tear_down() {
  bazel clean --expunge
  bazel shutdown
  rm -rf pkg
}

function do_sandbox_base_wiped_only_on_startup_test {
  local extra_args=( "${@}" )

  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(name = "pkg", outs = ["pkg.out"], cmd = "echo >\$@")
EOF

  local output_base="$(bazel info output_base)"

  do_build() {
    bazel build --genrule_strategy=sandboxed "${extra_args[@]}" //pkg
  }

  do_build >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
  find "${output_base}" >>"${TEST_log}" 2>&1 || true

  local sandbox_dir="$(echo "${output_base}/sandbox"/*-sandbox)"
  [[ -d "${sandbox_dir}" ]] \
    || fail "${sandbox_dir} is missing; prematurely deleted?"

  local garbage="${output_base}/sandbox/garbage"
  mkdir -p "${garbage}/some/nested/contents"
  do_build >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
  expect_not_log "Deleting stale sandbox"
  [[ -d "${garbage}" ]] \
    || fail "Spurious contents deleted from sandbox base too early"

  bazel shutdown
  do_build >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
  expect_log "Deleting stale sandbox"
  [[ ! -d "${garbage}" ]] \
    || fail "sandbox base was not cleaned on restart"
}

function test_sandbox_base_wiped_only_on_startup_with_sync_deletions() {
  do_sandbox_base_wiped_only_on_startup_test \
    --experimental_sandbox_async_tree_delete_idle_threads=0
}

function test_sandbox_base_wiped_only_on_startup_with_async_deletions() {
  do_sandbox_base_wiped_only_on_startup_test \
    --experimental_sandbox_async_tree_delete_idle_threads=HOST_CPUS
}

function do_succeed_when_executor_not_initialized_test() {
  local extra_args=( "${@}" )

  mkdir pkg
  mkfifo pkg/BUILD

  bazel build --spawn_strategy=sandboxed --nobuild "${@}" //pkg:all \
    >"${TEST_log}" 2>&1 &
  local pid="${!}"

  echo "Waiting for Blaze to finish initializing all modules"
  while ! grep "currently loading: pkg" "${TEST_log}"; do
    sleep 1
  done

  echo "Interrupting Blaze before it gets to init the executor"
  kill "${pid}"

  echo "And now giving Blaze a chance to finalize all modules"
  echo "unblock fifo" >pkg/BUILD
  wait "${pid}" || true

  expect_log "Build did NOT complete successfully"
  # Disallow some common messages we might see during a crash.
  expect_not_log "Internal error"
  expect_not_log "stack trace"
  expect_not_log "NullPointerException"
}

function test_succeed_when_executor_not_initialized_with_defaults() {
  # Pass a no-op flag to the test to workaround a bug in macOS's default
  # and ancient bash version which causes it to error out on an empty
  # argument list when $@ is consumed and set -u is enabled.
  local noop=( --nobuild )

  do_succeed_when_executor_not_initialized_test "${noop[@]}"
}

function test_succeed_when_executor_not_initialized_with_async_deletions() {
  do_succeed_when_executor_not_initialized_test \
    --experimental_sandbox_async_tree_delete_idle_threads=auto
}

function test_sandbox_base_can_be_rm_rfed() {
  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(name = "pkg", outs = ["pkg.out"], cmd = "echo >\$@")
EOF

  local output_base="$(bazel info output_base)"

  do_build() {
    bazel build --genrule_strategy=sandboxed //pkg
  }

  do_build >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
  find "${output_base}" >>"${TEST_log}" 2>&1 || true

  local sandbox_base="${output_base}/sandbox"
  [[ -d "${sandbox_base}" ]] \
    || fail "${sandbox_base} is missing; build did not use sandboxing?"

  # Ensure the sandbox base does not contain protected files that would prevent
  # a simple "rm -rf" from working under an unprivileged user.
  rm -rf "${sandbox_base}" || fail "Cannot clean sandbox base"

  # And now ensure Bazel reconstructs the sandbox base on a second build.
  do_build >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
}

function test_sandbox_old_contents_not_reused_in_consecutive_builds() {
  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(
    name = "pkg",
    srcs = ["pkg.in"],
    outs = ["pkg.out"],
    cmd = "cp \$(location :pkg.in) \$@",
)
EOF
  touch pkg/pkg.in

  for i in $(seq 5); do
    # Ensure that, even if we don't clean up the sandbox at all (with
    # --sandbox_debug), consecutive builds don't step on each other by trying to
    # reuse previous spawn identifiers.
    bazel build --genrule_strategy=sandboxed --sandbox_debug //pkg \
      >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
    echo foo >>pkg/pkg.in
  done
}

# Builds a target with the given strategy and ensures that the actions require
# params files to be written in the output base.
function build_with_params() {
  local strategy="${1}"; shift

  # Build a Java binary during this test because the Java rules work well with
  # sandboxing and support workers.
  mkdir pkg
  cat >pkg/BUILD <<'EOF'
java_binary(
    name = "java",
    srcs = ["Main.java"],
    main_class = "pkg.Main",
)
EOF
  cat >pkg/Main.java <<'EOF'
package pkg;
public class Main {
  public static void main(String[] args) {}
}
EOF

  bazel build \
    --strategy=Javac="${strategy}" \
    --strategy=JavaResourceJar="${strategy}" \
    --sandbox_debug \
    --min_param_file_size=100 \
    "${@}" \
    //pkg:java || fail "Build failed"
}

# Verifies that building a target that uses params files writes those params
# files to both the execroot and the sandbox.
function do_test_params_files_not_delayed() {
  local strategy="${1}"; shift

  local output_base
  output_base="$(bazel info output_base)" || fail "Cannot get output base"

  # Not passing --noexperimental_delay_virtual_input_materialization on
  # purpose to ensure that's the current default behavior.
  build_with_params "${strategy}" \
    --build  # Need a no-op flag to avoid set -u breakage on macOS.

  find -L "${output_base}" -name "*params" >files.txt || true
  grep -q "${output_base}/execroot" files.txt \
    || fail "Expected params files not found in execroot"
  grep -q "${output_base}/sandbox" files.txt \
    || fail "Expected params files not found in sandbox tree"
}

# Verifies that building a target that uses params files writes those params
# files only inside the sandbox when we delay virtual input artifact
# materialization.
function do_test_params_files_delayed() {
  local strategy="${1}"; shift

  local output_base
  output_base="$(bazel info output_base)" || fail "Cannot get output base"

  build_with_params "${strategy}" \
    --experimental_delay_virtual_input_materialization

  find -L "${output_base}" -name "*params" >files.txt || true
  grep -q "${output_base}/execroot" files.txt \
    && fail "Unexpected params files found in execroot"
  grep -q "${output_base}/sandbox" files.txt \
    || fail "Expected params files not found in sandbox tree"
}

# We expect "sandboxed" to use the system-specific sandbox instead of
# the processwrapper-sandbox (tested below). But if that's not the case,
# there is not much we can do here.
function test_params_files_not_delayed_default_sandbox() {
  do_test_params_files_not_delayed sandboxed
}
function test_params_files_delayed_default_sandbox() {
  do_test_params_files_delayed sandboxed
}

function test_params_files_not_delayed_process_wrapper_sandbox() {
  do_test_params_files_not_delayed processwrapper-sandbox
}
function test_params_files_delayed_process_wrapper_sandbox() {
  do_test_params_files_delayed processwrapper-sandbox
}

# Worker tests do not really belong in this file, but as we are exercising
# the same code path used for the sandbox regarding virtual input artifact
# materialization, we keep them here to reuse the testing logic.
function test_params_files_not_delayed_worker() {
  local output_base
  output_base="$(bazel info output_base)" || fail "Cannot get output base"

  # Not passing --noexperimental_delay_virtual_input_materialization on
  # purpose to ensure that's the current default behavior.
  build_with_params worker \
    --build  # Need a no-op flag to avoid set -u breakage on macOS.

  find -L "${output_base}" -name "*params" >files.txt || true
  grep -q "${output_base}/execroot" files.txt \
    || fail "Expected params files not found in execroot"
}
function test_params_files_delayed_worker() {
  local output_base
  output_base="$(bazel info output_base)" || fail "Cannot get output base"

  build_with_params worker \
    --experimental_delay_virtual_input_materialization

  find -L "${output_base}" -name "*params" >files.txt || true
  grep -q "${output_base}/execroot" files.txt \
    || fail "Expected params files not found in execroot"
}

run_suite "sandboxing"
