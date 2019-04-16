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

run_suite "sandboxing"
