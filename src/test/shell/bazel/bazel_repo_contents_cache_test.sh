#!/usr/bin/env bash
#
# Copyright 2025 The Bazel Authors. All rights reserved.
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
# Test repo_contents_cache behavior when declared within the WORKSPACE

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/remote_helpers.sh" \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }

function set_up() {
  write_default_bazelrc
  add_to_bazelrc "common --repo_contents_cache=.repo_contents_cache"
}

function tear_down() {
  shutdown_server
}

function test_repo_contents_cache_in_workspace_fails_without_being_bazelignored() {
  # If --repo_contents_cache flag is set and is pointing towards a directory
  # within the current workspace but said directory is not part of .bazelignore
  # bazel should exit.
  bazel version >& $TEST_log && echo "Expected the command to fail"
  expect_log "The repo contents cache \[.*\] is inside the main repo \[.*\]"

  # Post-test cleanup_workspace() calls "bazel clean", which would also fail
  # unless we reset the bazelrc.
  write_default_bazelrc
}

function test_repo_contents_cache_in_workspace_while_being_bazelignored() {
  # If --repo_contents_cache flag is set and is pointing towards a directory
  # within the current workspace and said directory is part of .bazelignore
  # bazel should run.

  echo ".repo_contents_cache" > .bazelignore
  bazel version >& $TEST_log || fail "Expected the command to succeed"
}

run_suite "repo_contents_cache tests"
