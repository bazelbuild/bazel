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

load("//pkg:build.bzl", "environ")

environ(name = "no_default_env", env = 0)
environ(name = "with_default_env", env = 1)

sh_test(
    name = "test_env_foo",
    srcs = ["test_env_foo.sh"],
)

genrule(
  name = "show_host_env_using_tools",
  tools = ["env.txt"],
  outs = ["tools_env.txt"],
  cmd = "cp \$(location env.txt) \"\$@\""
)

genrule(
  name = "show_host_env_using_successive_tools",
  tools = ["tools_env.txt"],
  outs = ["successive_tools_env.txt"],
  cmd = "cp \$(location tools_env.txt) \"\$@\""
)
EOF
  cat > pkg/build.bzl <<EOF
def _impl(ctx):
  output = ctx.outputs.out
  ctx.actions.run_shell(
      inputs=[],
      outputs=[output],
      use_default_shell_env = ctx.attr.env,
      command="env > %s" % output.path)

environ = rule(
    implementation=_impl,
    attrs={"env": attr.bool(default=True)},
    outputs={"out": "%{name}.env"},
)
EOF
  cat > pkg/test_env_foo.sh <<'EOF'
#!/bin/sh

echo "FOO is >${FOO}<"

{ echo "${FOO}" | grep foo; } || { echo "expected FOO to contain foo"; exit 1; }

EOF
  chmod u+x pkg/test_env_foo.sh
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
  bazel build --action_env=FOO -s pkg:showenv 2> $TEST_log \
      || fail "${PRODUCT_NAME} build showenv failed"
  expect_not_log '^SUBCOMMAND.*pkg:showenv'

  # However, if a used variable changes, we expect the change to be propagated
  export FOO=changed_foo
  bazel build --action_env=FOO -s pkg:showenv 2> $TEST_log \
      || fail "${PRODUCT_NAME} build showenv failed"
  expect_log '^SUBCOMMAND.*pkg:showenv'
  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/env.txt > $TEST_log
  expect_log "FOO=changed_foo"

  # But repeating the build with no further changes, no action should happen
  bazel build --action_env=FOO -s pkg:showenv 2> $TEST_log \
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
  # Recreate .bazelrc as removing it affects other tests that run in the
  # same shard with this test.
  write_default_bazelrc
}

function test_use_default_shell_env {
    bazel build --action_env=FOO=bar //pkg/...
    echo
    cat bazel-bin/pkg/with_default_env.env
    echo
    grep -q FOO=bar bazel-bin/pkg/with_default_env.env \
        || fail "static action environment not honored"
    (grep -q FOO=bar bazel-bin/pkg/no_default_env.env \
         && fail "static action_env used, even though requested not to") || true

    export BAR=baz
    bazel build --action_env=BAR //pkg/...
    grep -q BAR=baz bazel-bin/pkg/with_default_env.env \
        || fail "dynamic action environment not honored"
    (grep -q BAR bazel-bin/pkg/no_default_env.env \
         && fail "dynamic action_env used, even though requested not to") || true
}

function test_action_env_changes_honored {
    # Verify that changes to the explicitly specified action_env in honored in
    # tests. Regression test for #3265.

    # start with a fresh bazel, to have a reproducible starting point
    bazel clean --expunge
    bazel test --test_output=all --action_env=FOO=foo //pkg:test_env_foo \
        || fail "expected to pass with correct value for FOO"
    # While the test is cached, changing the environment should rerun it and
    # detect the failure in the new environment.
    (bazel test --test_output=all --action_env=FOO=bar //pkg:test_env_foo \
         && fail "expected to fail with incorrect value for FOO") || true
    # Redo the same FOO being taken from the environment
    env FOO=foo bazel test --test_output=all --action_env=FOO //pkg:test_env_foo \
        || fail "expected to pass with correct value for FOO from the environment"
    (env FOO=bar bazel test --test_output=all --action_env=FOO=bar //pkg:test_env_foo \
         && fail "expected to fail with incorrect value for FOO from the environment") || true

}

function test_host_env_using_tools_simple() {
  export FOO=baz

  # If FOO is passed using --host_action_env, it should be listed in host env vars
  bazel build --host_action_env=FOO=bar pkg:show_host_env_using_tools \
      || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_log "FOO=bar"

  # But if FOO is passed using --action_env, it should not be listed in host env vars
  bazel build --action_env=FOO=bar pkg:show_host_env_using_tools \
      || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_not_log "FOO=bar"
}

function test_host_env_using_tools_latest_wins() {
  export FOO=environmentfoo
  export BAR=environmentbar
  bazel build --host_action_env=FOO=foo \
      --host_action_env=BAR=willbeoverridden --host_action_env=BAR=bar pkg:show_host_env_using_tools \
      || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_log "FOO=foo"
  expect_log "BAR=bar"
}

function test_client_env_using_tools() {
  export FOO=startup_foo
  bazel clean --expunge
  bazel help build > /dev/null || fail "${PRODUCT_NAME} help failed"
  export FOO=client_foo
  bazel build --host_action_env=FOO pkg:show_host_env_using_tools || \
    fail "${PRODUCT_NAME} build show_host_env_using_tools failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_log "FOO=client_foo"
}

function test_redo_host_env_using_tools() {
  export FOO=initial_foo
  export UNRELATED=some_value
  bazel build --host_action_env=FOO pkg:show_host_env_using_tools \
    || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_log "FOO=initial_foo"

  # If an unrelated value changes, we expect the action not to be executed again
  export UNRELATED=some_other_value
  bazel build --host_action_env=FOO -s pkg:show_host_env_using_tools 2> $TEST_log \
      || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"
  expect_not_log '^SUBCOMMAND.*pkg:show_host_env_using_tools'

  # However, if a used variable changes, we expect the change to be propagated
  export FOO=changed_foo
  bazel build --host_action_env=FOO -s pkg:show_host_env_using_tools 2> $TEST_log \
      || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"
  expect_log '^SUBCOMMAND.*pkg:show_host_env_using_tools'
  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_log "FOO=changed_foo"

  # But repeating the build with no further changes, no action should happen
  bazel build --host_action_env=FOO -s pkg:show_host_env_using_tools 2> $TEST_log \
      || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"
  expect_not_log '^SUBCOMMAND.*pkg:show_host_env_using_tools'
}

function test_latest_wins_arg_using_tools() {
  export FOO=bar
  export BAR=baz
  bazel build --host_action_env=BAR --host_action_env=FOO --host_action_env=FOO=foo \
      pkg:show_host_env_using_tools || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_log "FOO=foo"
  expect_log "BAR=baz"
  expect_not_log "FOO=bar"
}

function test_latest_wins_env_using_tools() {
  export FOO=bar
  export BAR=baz
  bazel build --host_action_env=BAR --host_action_env=FOO=foo --host_action_env=FOO \
      pkg:show_host_env_using_tools || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_log "FOO=bar"
  expect_log "BAR=baz"
  expect_not_log "FOO=foo"
}

function test_host_env_using_tools_simple() {
  export FOO=baz

  # If FOO is passed using --host_action_env, it should be listed in host env vars
  bazel build --host_action_env=FOO=bar pkg:show_host_env_using_tools \
      || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_log "FOO=bar"

  # But if FOO is passed using --action_env, it should not be listed in host env vars
  bazel build --action_env=FOO=bar pkg:show_host_env_using_tools \
      || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_not_log "FOO=bar"
}

function test_host_env_using_tools_latest_wins() {
  export FOO=environmentfoo
  export BAR=environmentbar
  bazel build --host_action_env=FOO=foo \
      --host_action_env=BAR=willbeoverridden --host_action_env=BAR=bar pkg:show_host_env_using_tools \
      || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_log "FOO=foo"
  expect_log "BAR=bar"
}

function test_client_env_using_tools() {
  export FOO=startup_foo
  bazel clean --expunge
  bazel help build > /dev/null || fail "${PRODUCT_NAME} help failed"
  export FOO=client_foo
  bazel build --host_action_env=FOO pkg:show_host_env_using_tools || \
    fail "${PRODUCT_NAME} build show_host_env_using_tools failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_log "FOO=client_foo"
}

function test_redo_host_env_using_tools() {
  export FOO=initial_foo
  export UNRELATED=some_value
  bazel build --host_action_env=FOO pkg:show_host_env_using_tools \
    || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_log "FOO=initial_foo"

  # If an unrelated value changes, we expect the action not to be executed again
  export UNRELATED=some_other_value
  bazel build --host_action_env=FOO -s pkg:show_host_env_using_tools 2> $TEST_log \
      || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"
  expect_not_log '^SUBCOMMAND.*pkg:show_host_env_using_tools'

  # However, if a used variable changes, we expect the change to be propagated
  export FOO=changed_foo
  bazel build --host_action_env=FOO -s pkg:show_host_env_using_tools 2> $TEST_log \
      || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"
  expect_log '^SUBCOMMAND.*pkg:show_host_env_using_tools'
  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_log "FOO=changed_foo"

  # But repeating the build with no further changes, no action should happen
  bazel build --host_action_env=FOO -s pkg:show_host_env_using_tools 2> $TEST_log \
      || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"
  expect_not_log '^SUBCOMMAND.*pkg:show_host_env_using_tools'
}

function test_latest_wins_arg_using_tools() {
  export FOO=bar
  export BAR=baz
  bazel build --host_action_env=BAR --host_action_env=FOO --host_action_env=FOO=foo \
      pkg:show_host_env_using_tools || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_log "FOO=foo"
  expect_log "BAR=baz"
  expect_not_log "FOO=bar"
}

function test_latest_wins_env_using_tools() {
  export FOO=bar
  export BAR=baz
  bazel build --host_action_env=BAR --host_action_env=FOO=foo --host_action_env=FOO \
      pkg:show_host_env_using_tools || fail "${PRODUCT_NAME} build show_host_env_using_tools failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/tools_env.txt > $TEST_log
  expect_log "FOO=bar"
  expect_log "BAR=baz"
  expect_not_log "FOO=foo"
}

function test_host_env_using_successive_tools_simple() {
  export FOO=baz

  bazel build --host_action_env=FOO=bar pkg:show_host_env_using_successive_tools \
      || fail "${PRODUCT_NAME} build show_host_env_using_successive_tool failed"

  cat `bazel info ${PRODUCT_NAME}-genfiles`/pkg/successive_tools_env.txt > $TEST_log
  expect_log "FOO=bar"
}

run_suite "Tests for bazel's handling of environment variables in actions"
