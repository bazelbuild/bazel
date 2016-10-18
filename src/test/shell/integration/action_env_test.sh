#!/bin/bash
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
# An end-to-end test that Bazel's provides the correct environment variables
# to actions.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

#### SETUP #############################################################

set -e

function set_up() {
  mkdir -p pkg
  cat > pkg/BUILD <<EOF
genrule(
  name = "showenv",
  outs = ["env.txt"],
  cmd = "env | sort > \"\$@\""
)
EOF
}

#### TESTS #############################################################

function test_simple() {
  export FOO=baz
  bazel build --action_env=FOO=bar pkg:showenv \
      || fail "${PRODUCT_NAME} build showenv failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/env.txt > $TEST_log
  expect_log "FOO=bar"
}

function test_simple_latest_wins() {
  export FOO=environmentfoo
  export BAR=environmentbar
  bazel build --action_env=FOO=foo \
      --action_env=BAR=willbeoverridden --action_env=BAR=bar pkg:showenv \
      || fail "${PRODUCT_NAME} build showenv failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/env.txt > $TEST_log
  expect_log "FOO=foo"
  expect_log "BAR=bar"
}

function test_client_env() {
  export FOO=startup_foo
  bazel clean --expunge
  bazel help build > /dev/null || fail "${PRODUCT_NAME} help failed"
  export FOO=client_foo
  bazel build --action_env=FOO pkg:showenv || \
    fail "${PRODUCT_NAME} build showenv failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/env.txt > $TEST_log
  expect_log "FOO=client_foo"
}

function test_redo_action() {
  export FOO=initial_foo
  export UNRELATED=some_value
  bazel build --action_env=FOO pkg:showenv \
    || fail "${PRODUCT_NAME} build showenv failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/env.txt > $TEST_log
  expect_log "FOO=initial_foo"

  # If an unrelated value changes, we expect the action not to be executed again
  export UNRELATED=some_other_value
  bazel build --action_env=FOO -s --experimental_ui pkg:showenv 2> $TEST_log \
      || fail "${PRODUCT_NAME} build showenv failed"
  expect_not_log '^SUBCOMMAND.*pkg:showenv'

  # However, if a used variable changes, we expect the change to be propagated
  export FOO=changed_foo
  bazel build --action_env=FOO -s --experimental_ui pkg:showenv 2> $TEST_log \
      || fail "${PRODUCT_NAME} build showenv failed"
  expect_log '^SUBCOMMAND.*pkg:showenv'
  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/env.txt > $TEST_log
  expect_log "FOO=changed_foo"

  # But repeating the build with no further changes, no action should happen
  bazel build --action_env=FOO -s --experimental_ui pkg:showenv 2> $TEST_log \
      || fail "${PRODUCT_NAME} build showenv failed"
  expect_not_log '^SUBCOMMAND.*pkg:showenv'
}

function test_latest_wins_arg() {
  export FOO=bar
  export BAR=baz
  bazel build --action_env=BAR --action_env=FOO --action_env=FOO=foo \
      pkg:showenv || fail "${PRODUCT_NAME} build showenv failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/env.txt > $TEST_log
  expect_log "FOO=foo"
  expect_log "BAR=baz"
  expect_not_log "FOO=bar"
}

function test_latest_wins_env() {
  export FOO=bar
  export BAR=baz
  bazel build --action_env=BAR --action_env=FOO=foo --action_env=FOO \
      pkg:showenv || fail "${PRODUCT_NAME} build showenv failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/env.txt > $TEST_log
  expect_log "FOO=bar"
  expect_log "BAR=baz"
  expect_not_log "FOO=foo"
}

function test_env_freezing() {
  add_to_bazelrc "build --action_env=FREEZE_TEST_FOO"
  add_to_bazelrc "build --action_env=FREEZE_TEST_BAR=is_fixed"
  add_to_bazelrc "build --action_env=FREEZE_TEST_BAZ=will_be_overridden"
  add_to_bazelrc "build --action_env=FREEZE_TEST_BUILD"

  export FREEZE_TEST_FOO=client_foo
  export FREEZE_TEST_BAR=client_bar
  export FREEZE_TEST_BAZ=client_baz
  export FREEZE_TEST_BUILD=client_build

  bazel info --action_env=FREEZE_TEST_BAZ client-env > $TEST_log

  expect_log "build --action_env=FREEZE_TEST_FOO=client_foo"
  expect_not_log "FREEZE_TEST_BAR"
  expect_log "build --action_env=FREEZE_TEST_BAZ=client_baz"
  expect_log "build --action_env=FREEZE_TEST_BUILD=client_build"

  rm -f .${PRODUCT_NAME}rc
}

run_suite "Tests for bazel's handling of environment variables in actions"
