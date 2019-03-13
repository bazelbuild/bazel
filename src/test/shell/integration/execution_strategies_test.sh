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
# execution_phase_tests.sh: miscellaneous integration tests of Bazel for
# behaviors that affect the execution phase.
#

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

# Tests that you have to opt-in to list based strategy selection via an incompatible flag.
function test_incompatible_flag_required() {
  bazel build --spawn_strategy=worker,local --debug_print_action_contexts &> $TEST_log || true
  expect_log "incompatible_list_based_execution_strategy_selection was not enabled"
}

# Tests that you can set the spawn strategy flags to a list of strategies.
function test_multiple_strategies() {
  bazel build --incompatible_list_based_execution_strategy_selection \
      --spawn_strategy=worker,local --debug_print_action_contexts &> $TEST_log || fail
  # Can't test for exact strategy names here, because they differ between platforms and products.
  expect_log "\"\" = \[.*, .*\]"
}

# Tests that the hardcoded Worker strategies are not introduced with the new
# strategy selection
function test_no_worker_defaults() {
  bazel build --incompatible_list_based_execution_strategy_selection \
      --debug_print_action_contexts &> $TEST_log || fail
  # Can't test for exact strategy names here, because they differ between platforms and products.
  expect_not_log "\"Closure\""
  expect_not_log "\"DexBuilder\""
  expect_not_log "\"Javac\""
}

# Tests that Bazel catches an invalid strategy list that has an empty string as an element.
function test_empty_strategy_in_list_is_forbidden() {
  bazel build --incompatible_list_based_execution_strategy_selection \
      --spawn_strategy=worker,,local --debug_print_action_contexts &> $TEST_log || true
  expect_log "--spawn_strategy=worker,,local: Empty values are not allowed as part of this comma-separated list of options"
}

# Test that when you set a strategy to the empty string, it gets removed from the map of strategies
# and thus results in the default strategy being used (the one set via --spawn_strategy=).
function test_empty_strategy_means_default() {
  bazel build --incompatible_list_based_execution_strategy_selection \
      --spawn_strategy=worker,local --strategy=FooBar=local \
      --debug_print_action_contexts &> $TEST_log || fail
  expect_log "\"FooBar\" = "

  bazel build --incompatible_list_based_execution_strategy_selection \
      --spawn_strategy=worker,local --strategy=FooBar=local --strategy=FooBar= \
      --debug_print_action_contexts &> $TEST_log || fail
  expect_not_log "\"FooBar\" = "
}

run_suite "Tests for the execution strategy selection."
