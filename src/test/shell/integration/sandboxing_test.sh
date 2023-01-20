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
    bazel build --genrule_strategy=sandboxed --sandbox_debug \
      "${extra_args[@]}" //pkg
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
    bazel build --genrule_strategy=sandboxed --sandbox_debug //pkg
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

function test_sandbox_not_used_with_legacy_fallback() {
  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(name = "pkg", outs = ["pkg.out"], cmd = "pwd; echo >\$@",
  tags = ["no-sandbox"])
EOF

  local output_base="$(bazel info output_base)"
  local sandbox_base="${output_base}/sandbox"
  rm -rf ${sandbox_base}

  bazel build --genrule_strategy=sandboxed \
    --incompatible_legacy_local_fallback //pkg \
    >"${TEST_log}" 2>&1 || fail "Expected build to succeed"

  expect_not_log "${output_base}.*/sandbox/"
  expect_log "implicit fallback from sandbox to local"
}

function test_sandbox_local_not_used_without_legacy_fallback() {
  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(name = "pkg", outs = ["pkg.out"], cmd = "pwd; echo >\$@",
  tags = ["no-sandbox"])
EOF

  local output_base="$(bazel info output_base)"
  local sandbox_base="${output_base}/sandbox"
  rm -rf ${sandbox_base}

  bazel build --genrule_strategy=sandboxed \
    --noincompatible_legacy_local_fallback //pkg \
    >"${TEST_log}" 2>&1 && fail "Expected build to fail" || true
  # Still warning in this case even when the flag is flipped
  expect_log "implicit fallback from sandbox to local"
}

function test_sandbox_local_used_with_proper_strategy() {
  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(name = "pkg", outs = ["pkg.out"], cmd = "pwd; echo >\$@",
  tags = ["no-sandbox"])
EOF

  local output_base="$(bazel info output_base)"
  local sandbox_base="${output_base}/sandbox"
  rm -rf ${sandbox_base}

  bazel build --genrule_strategy=sandboxed,standalone \
    --noincompatible_legacy_local_fallback //pkg \
    >"${TEST_log}" 2>&1 || fail "Expected build to succeed"

  expect_not_log "${output_base}.*/sandbox/"
  expect_not_log "implicit fallback from sandbox to local"
}

function test_sandbox_base_top_is_removed() {
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
  [[ ! -d "${sandbox_base}" ]] \
    || fail "${sandbox_base} left behind unnecessarily"

  # Restart Bazel and check we don't print spurious "Deleting stale sandbox"
  # warnings.
  bazel shutdown
  do_build >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
  expect_not_log "Deleting stale sandbox"
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

function test_sandbox_hardening_with_cgroups_v1() {
  if ! grep -E '^cgroup +[^ ]+ +cgroup +.*memory.*' /proc/mounts; then
    echo "No cgroup v1 memory controller mounted, skipping test"
    return 0
  fi
  memmount=$(grep -E '^cgroup +[^ ]+ +cgroup +.*memory.*' /proc/mounts | cut -d' ' -f2)
  if ! grep -E '^[0-9]*:[^:]*memory[^:]*:' /proc/self/cgroup &>/dev/null; then
    echo "Does not use cgroups v1, skipping test"
    return 0
  fi
  memsubdir=$(grep -E '^[0-9]*:[^:]*memory[^:]*:' /proc/self/cgroup | cut -d: -f3)
  memdir="$memmount$memsubdir"
  if [[ ! -w "$memdir" ]]; then
    echo "Cgroups v1 directory not writable, skipping test"
    return 0
  fi

  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(name = "pkg", outs = ["pkg.out"], cmd = "pwd; echo >\$@")
EOF
  local genfiles_base="$(bazel info ${PRODUCT_NAME}-genfiles)"

  bazel build --genrule_strategy=linux-sandbox \
    --experimental_sandbox_memory_limit=1000000 //pkg \
    >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
  rm -f ${genfiles_base}/pkg/pkg.out
  bazel build --genrule_strategy=linux-sandbox \
    --experimental_sandbox_memory_limit=100 //pkg \
    >"${TEST_log}" 2>&1 && fail "Expected build to fail" || true
}

function test_sandbox_hardening_with_cgroups_v2() {
  if ! grep -E '^cgroup2 +[^ ]+ +cgroup2 ' /proc/mounts; then
    echo "No cgroup2 mounted, skipping test"
    return 0
  fi
  if ! grep -E '^0::' /proc/self/cgroup &>/dev/null; then
    echo "Does not use cgroups v2, skipping test"
    return 0
  fi
  if ! XDG_RUNTIME_DIR=/run/user/$( id -u ) systemd-run --user --scope true; then
    echo "Not able to use systemd, skipping test"
    return 0
  fi

  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(name = "pkg", outs = ["pkg.out"], cmd = "pwd; echo >\$@")
EOF
  local genfiles_base="$(bazel info ${PRODUCT_NAME}-genfiles)"
  # Need to make sure the bazel server runs under systemd, too.
  bazel shutdown

  XDG_RUNTIME_DIR=/run/user/$( id -u ) systemd-run --user --scope \
  bazel build --genrule_strategy=linux-sandbox \
    --experimental_sandbox_memory_limit=1000000 //pkg \
    >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
  rm -f ${genfiles_base}/pkg/pkg.out

  bazel shutdown
  XDG_RUNTIME_DIR=/run/user/$( id -u ) systemd-run --user --scope \
  bazel build --genrule_strategy=linux-sandbox \
    --experimental_sandbox_memory_limit=100 //pkg \
    >"${TEST_log}" 2>&1 && fail "Expected build to fail" || true
}

run_suite "sandboxing"
