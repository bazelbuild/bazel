#!/usr/bin/env bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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
# Test of Bazel's startup option handling.

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

# AOT tests create and corrupt install-base-wide caches. Never use the install
# base shared by concurrently running tests on CI; let the test harness use
# its private output user root instead.
unset TEST_INSTALL_BASE

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_different_startup_options() {
  pid=$(bazel --nobatch info server_pid 2> $TEST_log)
  [[ -n $pid ]] || fail "Couldn't run ${PRODUCT_NAME}"
  newpid=$(bazel --batch --host_jvm_args=-Xmx4321m info server_pid 2> $TEST_log)
  expect_log "WARNING: Running B\\(azel\\|laze\\) server needs to be killed, because the following startup options are different:
  - Only in old server: --noshutdown_on_low_sys_mem
  - Only in new server: --batch --host_jvm_args=-Xmx4321m"
  [[ "$newpid" != "$pid" ]] || fail "pid $pid was the same!"
  if ! is_windows; then
    # On Windows: the kill command of MSYS doesn't work for Windows PIDs.
    kill -0 $pid 2> /dev/null && fail "$pid not dead" || true
    kill -0 $newpid 2> /dev/null && fail "$newpid not dead" || true
  fi
}

# Regression test for Issue #1659
function test_command_args_are_not_parsed_as_startup_args() {
  bazel info --bazelrc=bar &> $TEST_log && fail "Should fail"
  expect_log "Unrecognized option: --bazelrc=bar"
  expect_not_log "Error: Unable to read .bazelrc file"
}

# Test that normal bazel works with and without --autodetect_server_javabase
# because it has an embedded JRE.
function test_autodetect_server_javabase() {
  bazel --autodetect_server_javabase version &> $TEST_log || fail "Should pass"
  bazel --noautodetect_server_javabase version &> $TEST_log || fail "Should pass"
}

# Below are the regression tests for Issue #7489
function test_multiple_bazelrc_later_overwrites_earlier() {
  # Help message only visible with --help_verbosity=medium
  help_message_in_description="--${PRODUCT_NAME}rc (a string; may be used multiple times)"

  echo "help --help_verbosity=short" > 1.rc
  echo "help --help_verbosity=medium" > 2.rc
  bazel "--${PRODUCT_NAME}rc=1.rc" "--${PRODUCT_NAME}rc=2.rc" help startup_options &> $TEST_log || fail "Should pass"
  expect_log "$help_message_in_description"

  echo "help --help_verbosity=medium" > 1.rc
  echo "help --help_verbosity=short" > 2.rc
  bazel "--${PRODUCT_NAME}rc=1.rc" "--${PRODUCT_NAME}rc=2.rc" help startup_options &> $TEST_log || fail "Should pass"
  expect_not_log "$help_message_in_description"
}

function test_multiple_bazelrc_set_different_options() {
  # Set host platform to avoid using the default value which only works for Bazel.
  echo "common --host_platform=${default_host_platform}" > host_platform.rc
  echo "common --verbose_failures" > 1.rc
  echo "common --test_output=all" > 2.rc
  bazel "--${PRODUCT_NAME}rc=host_platform.rc" "--${PRODUCT_NAME}rc=1.rc" "--${PRODUCT_NAME}rc=2.rc" info --announce_rc &> $TEST_log || fail "Should pass"
  expect_log "Inherited 'common' options: --verbose_failures"
  expect_log "Inherited 'common' options: --test_output=all"
}

function test_bazelrc_after_devnull_ignored() {
  # Set host platform to avoid using the default value which only works for Bazel.
  echo "common --host_platform=${default_host_platform}" > host_platform.rc
  echo "common --verbose_failures" > 1.rc
  echo "common --test_output=all" > 2.rc
  echo "common --definitely_invalid_config" > 3.rc

  bazel "--${PRODUCT_NAME}rc=host_platform.rc" "--${PRODUCT_NAME}rc=1.rc" "--${PRODUCT_NAME}rc=2.rc" "--${PRODUCT_NAME}rc=/dev/null" \
   "--${PRODUCT_NAME}rc=3.rc" info --announce_rc &> $TEST_log || fail "Should pass"
  expect_log "Inherited 'common' options: --verbose_failures"
  expect_log "Inherited 'common' options: --test_output=all"
  expect_not_log "--definitely_invalid_config"
}

function test_experimental_aot_cache_training_run() {
  local install_base
  install_base=$(bazel info install_base 2> $TEST_log) \
    || fail "Couldn't run ${PRODUCT_NAME}"
  local aot_cache="${install_base}.aot"
  rm -f "$aot_cache" "$aot_cache".*

  mkdir -p pkg
  cat > pkg/BUILD <<'EOF'
genrule(name = "gen", outs = ["gen.txt"], cmd = "touch $@")
EOF

  # A training run starts a new server, which records the cache until it is
  # shut down. Invocations without the option keep using that server.
  local training_pid pid
  training_pid=$(bazel --experimental_aot_cache_training_run info server_pid \
    2> $TEST_log) || fail "Couldn't start the training run"
  bazel build //pkg:gen &> $TEST_log \
    || fail "Couldn't build in the training run"
  expect_not_log "server needs to be killed"
  pid=$(bazel info server_pid 2> $TEST_log) \
    || fail "Couldn't run ${PRODUCT_NAME}"
  expect_not_log "server needs to be killed"
  assert_equals "$training_pid" "$pid"
  [[ -e "$aot_cache" ]] && fail "AOT cache assembled before the server exited"
  bazel shutdown &> $TEST_log || fail "Couldn't shut down the server"
  expect_not_log "server needs to be killed"
  [[ -s "$aot_cache" ]] || fail "AOT cache not assembled at $aot_cache"

  # The cache is used by servers started from now on. -XX:AOTMode=on makes the
  # JVM fail to start instead of silently ignoring a cache it can't use.
  bazel shutdown &> $TEST_log || fail "Couldn't shut down the server"
  bazel --host_jvm_args=-XX:AOTMode=on info server_pid &> $TEST_log \
    || fail "Couldn't start the server with the AOT cache"
  bazel --host_jvm_args=-XX:AOTMode=on shutdown &> $TEST_log \
    || fail "Couldn't shut down the server"
  rm -f "$aot_cache"
}

function test_experimental_aot_cache_training_run_restarts_server() {
  local install_base
  install_base=$(bazel info install_base 2> $TEST_log) \
    || fail "Couldn't run ${PRODUCT_NAME}"
  local aot_cache="${install_base}.aot"
  rm -f "$aot_cache" "$aot_cache".*

  # A training run restarts a running server even if its startup options are
  # the same, which includes a server that is already recording a cache.
  local pid1 pid2 pid3
  pid1=$(bazel info server_pid 2> $TEST_log) \
    || fail "Couldn't run ${PRODUCT_NAME}"
  pid2=$(bazel --experimental_aot_cache_training_run info server_pid \
    2> $TEST_log) || fail "Couldn't start the training run"
  expect_log "server needs to be killed, because --experimental_aot_cache"
  assert_not_equals "$pid1" "$pid2"
  [[ -e "$aot_cache" ]] && fail "AOT cache assembled by a non-recording server"

  pid3=$(bazel --experimental_aot_cache_training_run info server_pid \
    2> $TEST_log) || fail "Couldn't start the second training run"
  expect_log "server needs to be killed, because --experimental_aot_cache"
  assert_not_equals "$pid2" "$pid3"
  # The first training run ended when its server was restarted.
  [[ -s "$aot_cache" ]] || fail "AOT cache not assembled at $aot_cache"

  bazel shutdown &> $TEST_log || fail "Couldn't shut down the server"
  rm -f "$aot_cache"
}

function test_experimental_aot_cache_training_run_disabled_after_crash() {
  local install_base
  install_base=$(bazel info install_base 2> $TEST_log) \
    || fail "Couldn't run ${PRODUCT_NAME}"
  local aot_cache="${install_base}.aot"
  rm -f "$aot_cache" "$aot_cache".*

  # A cache that the JVM can't load is fatal with -XX:AOTMode=on (and in some
  # cases even without it). The client then deletes the cache and stops using
  # one for this install base until a new one is recorded.
  echo "not an AOT cache" > "$aot_cache"
  bazel --host_jvm_args=-XX:AOTMode=on info server_pid &> $TEST_log \
    && fail "Server started with an unusable AOT cache"
  expect_log "Server crashed during startup"
  expect_log "The cache has been deleted"
  [[ -e "$aot_cache" ]] && fail "Unusable AOT cache not deleted"
  [[ -e "$aot_cache.disabled" ]] || fail "AOT cache not disabled"

  bazel info server_pid &> $TEST_log \
    || fail "Couldn't start the server with the AOT cache disabled"

  # A training run records a new cache and re-enables its use.
  bazel --experimental_aot_cache_training_run info server_pid &> $TEST_log \
    || fail "Couldn't start the training run"
  [[ -e "$aot_cache.disabled" ]] && fail "AOT cache still disabled"
  bazel shutdown &> $TEST_log || fail "Couldn't shut down the server"
  [[ -s "$aot_cache" ]] || fail "AOT cache not assembled at $aot_cache"
  rm -f "$aot_cache"
}

function test_experimental_aot_cache_batch_mode() {
  local install_base output_base actual
  install_base=$(bazel info install_base 2> "$TEST_log") \
    || fail "Couldn't get install base"
  output_base=$(bazel info output_base 2> "$TEST_log") \
    || fail "Couldn't get output base"
  local aot_cache="${install_base}.aot"
  rm -f "$aot_cache" "$aot_cache".*

  bazel --experimental_aot_cache_training_run info server_pid &> "$TEST_log" \
    || fail "Couldn't start the training run"
  bazel shutdown &> "$TEST_log" || fail "Couldn't shut down the server"
  [[ -s "$aot_cache" ]] || fail "AOT cache not assembled"

  # An incompatible cache must not add JVM diagnostics to batch stdout.
  actual=$(bazel --batch --host_jvm_args=-XX:-UseCompactObjectHeaders \
    info output_base 2> "$TEST_log") || fail "Couldn't run in batch mode"
  assert_equals "$output_base" "$actual"

  # -Xshare:off and -XX:AOTCache cannot be used together. Batch mode must
  # continue to work without relying on the client/server crash recovery.
  local attempt
  for attempt in 1 2; do
    actual=$(bazel --batch --host_jvm_args=-Xshare:off info output_base \
      2> "$TEST_log") || fail "Batch invocation $attempt failed"
    assert_equals "$output_base" "$actual"
  done

  echo "not an AOT cache" > "$aot_cache"
  actual=$(bazel --batch info output_base 2> "$TEST_log") \
    || fail "Couldn't run in batch mode with an unusable cache"
  assert_equals "$output_base" "$actual"
  [[ -e "$aot_cache.disabled" ]] && fail "Batch mode disabled the server cache"
  rm -f "$aot_cache"

  # Training also emits JVM diagnostics on stdout, so it is disabled in
  # batch mode even when explicitly requested.
  actual=$(bazel --batch --experimental_aot_cache_training_run info output_base \
    2> "$TEST_log") || fail "Couldn't run in batch mode with training requested"
  assert_equals "$output_base" "$actual"
  [[ ! -e "$aot_cache" ]] || fail "Batch mode recorded a cache"
}

function test_experimental_aot_cache_concurrent_training_runs() {
  local install_base
  install_base=$(bazel info install_base 2> "$TEST_log") \
    || fail "Couldn't get install base"
  local aot_cache="${install_base}.aot"
  rm -f "$aot_cache" "$aot_cache".*

  local output_base1="${TEST_TMPDIR}/aot_output1"
  local output_base2="${TEST_TMPDIR}/aot_output2"
  local pid1 pid2
  pid1=$(bazel --output_base="$output_base1" \
    --experimental_aot_cache_training_run info server_pid 2> "$TEST_log") \
    || fail "Couldn't start the first training run"
  pid2=$(bazel --output_base="$output_base2" \
    --experimental_aot_cache_training_run info server_pid 2> "$TEST_log") \
    || fail "Couldn't start the second training run"

  bazel --output_base="$output_base1" shutdown > "${TEST_TMPDIR}/shutdown1.log" 2>&1 &
  local shutdown1=$!
  bazel --output_base="$output_base2" shutdown > "${TEST_TMPDIR}/shutdown2.log" 2>&1 &
  local shutdown2=$!
  local shutdown1_status=0 shutdown2_status=0
  wait "$shutdown1" || shutdown1_status=$?
  wait "$shutdown2" || shutdown2_status=$?
  assert_equals 0 "$shutdown1_status"
  assert_equals 0 "$shutdown2_status"

  # Each JVM records a separate intermediate configuration, even though both
  # produce the same shared cache. Check both assembly processes succeeded.
  cat "$output_base1/server/jvm.out" > "$TEST_log"
  expect_log "AOTConfiguration recorded: .*\\.pid${pid1}\\.config"
  expect_log "AOTCache creation is complete:"
  expect_not_log "Child process failed"
  cat "$output_base2/server/jvm.out" > "$TEST_log"
  expect_log "AOTConfiguration recorded: .*\\.pid${pid2}\\.config"
  expect_log "AOTCache creation is complete:"
  expect_not_log "Child process failed"
  [[ -e "${aot_cache}.pid${pid1}.config" ]] && fail "First configuration not deleted"
  [[ -e "${aot_cache}.pid${pid2}.config" ]] && fail "Second configuration not deleted"
  [[ -s "$aot_cache" ]] || fail "AOT cache not assembled"
  rm -f "$aot_cache"
}

run_suite "${PRODUCT_NAME} startup options test"
