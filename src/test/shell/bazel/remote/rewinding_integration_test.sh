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
#
# Tests Bazel's recovery from lost remote CAS entries.
#
# These tests model the real-world scenario of a warm remote cache that has
# evicted part of its contents and, being under pressure, keeps dropping blobs
# as they are re-uploaded. The setup, shared by all tests, is:
#
#   1. Build a synthetic, generated target structure (a deep chain of "library"
#      genrules feeding a top-level genrule) with remote execution and build
#      without the bytes, populating the remote cache. Intermediate outputs are
#      not downloaded, so they live only remotely.
#   2. Restart the worker so that it evicts the entire remote CAS (keeping the
#      action cache, via --evict_existing_percentage) and keeps dropping blobs
#      as they are re-uploaded (via --lost_blob_percentage), modelling a cache
#      that lost its contents and is still losing freshly produced ones.
#   3. Modify the top-level target's own source so that only the top-level
#      action is invalidated. Its transitive dependencies are unchanged and so
#      are not re-run; their now-evicted outputs become lost inputs when the
#      re-running top-level action requests them.
#
# The phase-3 build is then run with three different recovery mechanisms to
# demonstrate the recovery hierarchy:
#
#   * No recovery: the build fails on the first lost input.
#   * Whole-build retries (--experimental_remote_cache_eviction_retries): a
#     whole-build retry regenerates the lost outputs, but because the cache
#     immediately drops each regenerated blob again, only the deepest
#     not-yet-restored level makes permanent progress per retry. A chain deeper
#     than the retry budget therefore still fails. (A single one-time eviction,
#     by contrast, is fully recovered by one retry; the ongoing loss is what
#     defeats whole-build retries.)
#   * Action rewinding (--rewind_lost_inputs): the lost inputs are regenerated
#     within a single build by repeatedly rewinding and re-executing the
#     generating actions, so the build succeeds.
#
# A separate test covers the case whole-build retries *are* designed for: a
# substantial one-time eviction with no ongoing loss is recovered by a single
# whole-build retry.
#
# The genrule commands are silent, so their stdout/stderr blobs are empty and
# hence never lost; combined with losing each blob only once per upload, this
# keeps the number of transient remote failures per action (which share the
# --remote_retries budget) small.

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
source "$(rlocation "io_bazel/src/test/shell/bazel/remote_helpers.sh")" \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }
source "$(rlocation "io_bazel/src/test/shell/bazel/remote/remote_utils.sh")" \
  || { echo "remote_utils.sh not found!" >&2; exit 1; }

# Number of "library" genrules in the generated dependency chain. The chain must
# be deeper than EVICTION_RETRIES: a whole-build retry only regenerates one more
# level of the lost chain per attempt, so a chain this deep cannot be recovered
# within the retry budget (see test_build_rewinding_is_insufficient).
readonly NUM_LIBS=10
readonly EVICTION_RETRIES=5

function set_up() {
  start_remote_worker
}

function tear_down() {
  bazel clean >& $TEST_log
  stop_worker
}

# Like start_worker from remote_utils.sh, but redirects the worker's output to a
# dedicated log file (instead of $TEST_log, which is repeatedly truncated by the
# bazel invocations under test) so that we can assert on the worker's eviction
# messages. Extra arguments are passed on to the worker.
function start_remote_worker() {
  work_path="${TEST_TMPDIR}/remote.work_path"
  cas_path="${TEST_TMPDIR}/remote.cas_path"
  pid_file="${TEST_TMPDIR}/remote.pid_file"
  worker_log="${TEST_TMPDIR}/remote.worker_log"
  mkdir -p "${work_path}"
  mkdir -p "${cas_path}"
  worker_port=$(pick_random_unused_tcp_port) || fail "no port found"
  local native_lib="${BAZEL_RUNFILES}/src/main/native/"
  "${REMOTE_WORKER}" \
      --singlejar \
      --jvm_flag=-Djava.library.path="${native_lib}" \
      --work_path="${work_path}" \
      --cas_path="${cas_path}" \
      --listen_port=${worker_port} \
      --pid_file="${pid_file}" \
      "$@" >& "${worker_log}" &
  wait_for_worker
}

# Restarts the worker against the same on-disk cache (so that an
# already-populated cache is preserved) on a fresh port, passing on any extra
# arguments. Output is appended to the worker log.
function restart_remote_worker() {
  if [ -s "${pid_file}" ]; then
    kill -9 "$(cat "${pid_file}")" || true
    rm -f "${pid_file}"
  fi
  worker_port=$(pick_random_unused_tcp_port) || fail "no port found"
  local native_lib="${BAZEL_RUNFILES}/src/main/native/"
  "${REMOTE_WORKER}" \
      --singlejar \
      --jvm_flag=-Djava.library.path="${native_lib}" \
      --work_path="${work_path}" \
      --cas_path="${cas_path}" \
      --listen_port=${worker_port} \
      --pid_file="${pid_file}" \
      "$@" >> "${worker_log}" 2>&1 &
  wait_for_worker
}

function wait_for_worker() {
  local wait_seconds=0
  until [[ -s "${pid_file}" || "$wait_seconds" -eq 30 ]]; do
    sleep 1
    wait_seconds=$((${wait_seconds} + 1))
  done
  if [ ! -s "${pid_file}" ]; then
    fail "Timed out waiting for remote worker to start."
  fi
}

# Generates a synthetic but somewhat realistic target structure in package pkg:
# a chain of NUM_LIBS "library" genrules (each depending on the previous one and
# on its own source file), topped by a "top" genrule that depends on the last
# library and on its own source main.txt. Modifying main.txt invalidates only
# the top genrule, turning the unchanged (and, in these tests, evicted) library
# outputs into lost inputs.
function setup_synthetic_workspace() {
  mkdir -p pkg
  local build_file="pkg/BUILD"
  : > "${build_file}"

  echo "BASE_CONTENT" > pkg/lib_0.txt
  cat >> "${build_file}" <<EOF
genrule(
    name = "lib_0",
    srcs = ["lib_0.txt"],
    outs = ["lib_0.out"],
    cmd = "cat \$(SRCS) > \$@",
)
EOF

  local i
  for ((i = 1; i < NUM_LIBS; i++)); do
    echo "lib ${i}" > "pkg/lib_${i}.txt"
    cat >> "${build_file}" <<EOF
genrule(
    name = "lib_${i}",
    srcs = ["lib_$((i - 1)).out", "lib_${i}.txt"],
    outs = ["lib_${i}.out"],
    cmd = "cat \$(SRCS) > \$@",
)
EOF
  done

  echo "main" > pkg/main.txt
  cat >> "${build_file}" <<EOF
genrule(
    name = "top",
    srcs = ["lib_$((NUM_LIBS - 1)).out", "main.txt"],
    outs = ["top.out"],
    cmd = "cat \$(SRCS) > \$@",
)
EOF
}

# Populates the remote cache with a clean build, then restarts the worker with
# the caller-specified loss behavior (passed on as extra worker flags, e.g.
# --evict_existing_percentage and/or --lost_blob_percentage) and modifies the
# top-level source. This leaves the workspace ready for a phase-3 build whose
# only way to make progress is to regenerate the lost, unchanged library
# outputs.
function populate_evict_and_modify() {
  setup_synthetic_workspace

  # Phase 1: populate the remote cache. With build without the bytes, the
  # library outputs are uploaded but not downloaded, so they live only remotely.
  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_download_toplevel \
      //pkg:top >& $TEST_log || fail "Failed to populate the remote cache"

  # Phase 2: restart the worker so that it loses cache contents as requested by
  # the caller. The restarted worker reuses the same on-disk cache, so the
  # action cache still references the now-missing blobs.
  restart_remote_worker --lost_blob_seed=rewinding_integration_test "$@"
  assert_cache_evicted

  # Phase 3 trigger: invalidate only the top-level action. Its dependencies are
  # unchanged and are not re-run, so their evicted outputs become lost inputs.
  echo "modified main" > pkg/main.txt
}

# Asserts that the worker actually evicted CAS entries on startup, so that a
# subsequent successful build genuinely exercised recovery. The worker logs the
# eviction before it starts serving and thus before it writes its pid file, which
# restart_remote_worker waits for, so the log line is present by the time this is
# called.
function assert_cache_evicted() {
  local evicted
  evicted=$(grep -oE "Evicted [0-9]+ existing CAS entries" "${worker_log}" \
      | grep -oE "[0-9]+" | tail -n 1 || true)
  if [[ -z "${evicted}" || "${evicted}" -lt 1 ]]; then
    cat "${worker_log}" >> $TEST_log
    fail "Expected the worker to evict at least one CAS entry"
  fi
}

# Asserts that the worker lost at least one blob right after an upload during the
# build, i.e. that the ongoing loss that defeats whole-build retries is in
# effect.
function assert_blobs_lost() {
  local losses
  losses=$(grep -c "Simulated loss of CAS entry" "${worker_log}" || true)
  if [[ "${losses}" -lt 1 ]]; then
    cat "${worker_log}" >> $TEST_log
    fail "Expected the worker to lose at least one blob during the build"
  fi
}

# Without any recovery mechanism, the build fails as soon as the re-running
# top-level action observes that one of its inputs was lost.
function test_no_recovery_fails() {
  populate_evict_and_modify --evict_existing_percentage=100 \
      --lost_blob_percentage=100

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_download_toplevel \
      --remote_retries=10 \
      --experimental_remote_cache_eviction_retries=0 \
      //pkg:top >& $TEST_log && fail "Expected build to fail without recovery"

  expect_log "pass --rewind_lost_inputs to enable recovery"
}

# Whole-build retries regenerate lost outputs, but each retry only advances one
# level down the chain of lost dependencies (the next-deeper output is only
# requested once the level above it has been regenerated). Since the chain is
# deeper than the retry budget, the build still fails after exhausting all
# retries.
function test_build_rewinding_is_insufficient() {
  populate_evict_and_modify --evict_existing_percentage=100 \
      --lost_blob_percentage=100

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_download_toplevel \
      --remote_retries=10 \
      --experimental_remote_cache_eviction_retries=${EVICTION_RETRIES} \
      //pkg:top >& $TEST_log \
      && fail "Expected build to fail despite whole-build retries"

  expect_log "Lost .* no longer available remotely"
  # The build should have exhausted exactly its retry budget before giving up.
  local retries
  retries=$(grep -c "Found transient remote cache error, retrying the build..." \
      $TEST_log || true)
  if [[ "${retries}" -ne "${EVICTION_RETRIES}" ]]; then
    fail "Expected ${EVICTION_RETRIES} whole-build retries, but observed ${retries}"
  fi
}

# Action rewinding regenerates the lost inputs within a single build by rewinding
# and re-executing the generating actions, so the build succeeds even though the
# same scenario defeats whole-build retries.
function test_action_rewinding_recovers() {
  populate_evict_and_modify --evict_existing_percentage=100 \
      --lost_blob_percentage=100

  local profile="${TEST_TMPDIR}/rewinding_profile.json"
  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_download_toplevel \
      --remote_retries=10 \
      --rewind_lost_inputs \
      --experimental_remote_cache_eviction_retries=0 \
      --profile="${profile}" \
      //pkg:top >& $TEST_log || fail "Expected build to succeed via action rewinding"

  # The worker actually kept losing blobs during the recovery build, so recovery
  # was non-trivial.
  assert_blobs_lost

  # The top-level output was downloaded and is correct, i.e. the base content
  # propagated all the way through the regenerated chain.
  [[ -f bazel-bin/pkg/top.out ]] \
      || fail "Expected top-level output bazel-bin/pkg/top.out to be downloaded"
  assert_contains "BASE_CONTENT" bazel-bin/pkg/top.out

  # Recovery happened via action rewinding, not via whole-build retries: the
  # profile records rewind plans and no whole-build retry was triggered.
  if ! grep -q "Preparing rewind plan" "${profile}"; then
    fail "Expected action rewind events in the profile ${profile}"
  fi
  expect_not_log "Found transient remote cache error"
}

# A substantial *one-time* loss of CAS entries (eviction without ongoing loss)
# is, unlike the ongoing loss above, recovered by a single whole-build retry:
# the retry invalidates all of the missing outputs at once and regenerates the
# entire chain in one evaluation, and since nothing is dropped again, the
# regenerated blobs stick. This is the case that whole-build retries are
# designed for.
function test_build_rewinding_recovers_from_one_time_eviction() {
  populate_evict_and_modify --evict_existing_percentage=100

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_download_toplevel \
      --remote_retries=10 \
      --experimental_remote_cache_eviction_retries=1 \
      //pkg:top >& $TEST_log \
      || fail "Expected build to succeed via a single whole-build retry"

  expect_log "Lost .* no longer available remotely"
  # A single whole-build retry was enough to recover.
  local retries
  retries=$(grep -c "Found transient remote cache error, retrying the build..." \
      $TEST_log || true)
  if [[ "${retries}" -ne 1 ]]; then
    fail "Expected exactly 1 whole-build retry, but observed ${retries}"
  fi

  # The top-level output was downloaded and is correct.
  [[ -f bazel-bin/pkg/top.out ]] \
      || fail "Expected top-level output bazel-bin/pkg/top.out to be downloaded"
  assert_contains "BASE_CONTENT" bazel-bin/pkg/top.out
}

run_suite "Tests for recovery from lost remote CAS entries"
