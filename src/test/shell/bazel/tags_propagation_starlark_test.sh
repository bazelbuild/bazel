#!/usr/bin/env bash
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

# Tests target's tags propagation with rules defined in Starlark.
# Tests for https://github.com/bazelbuild/bazel/issues/7766

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# Test a basic Starlark ctx.actions.run_shell rule which has tags, that should
# be propagated
function test_tags_propagated_to_run_shell() {
  mkdir -p test
  cat << EOF >> test/BUILD
load(":starlark.bzl", "test_rule")

test_rule(
    name = "test",
    out = "output.txt",
    tags = ["no-cache", "no-remote", "local"]
)
EOF

  cat << 'EOF' >> test/starlark.bzl
def _test_impl(ctx):
  ctx.actions.run_shell(outputs = [ctx.outputs.out],
                        command = "touch" + ctx.outputs.out.path)
  files_to_build = depset([ctx.outputs.out])
  return DefaultInfo(
      files = files_to_build,
  )

test_rule = rule(
    implementation=_test_impl,
    attrs = {
        "out": attr.output(mandatory = True),
    },
)
EOF

  bazel aquery --experimental_allow_tags_propagation '//test:test' > output1 2> $TEST_log \
      || fail "should have generated output successfully"

  assert_contains "ExecutionInfo: {local: '', no-cache: '', no-remote: ''}" output1
}

# Test a basic Starlark ctx.actions.run rule which has tags, that should be
# propagated.
function test_tags_propagated_to_run() {
  mkdir -p test
  cat << EOF >> test/BUILD
load(":starlark.bzl", "test_rule")

test_rule(
    name = "test",
    out = "output.txt",
    tags = ["no-cache", "no-remote", "no-sandbox", "requires-network", "local"]
)
EOF

  cat << 'EOF' >> test/starlark.bzl
def _test_impl(ctx):
  ctx.actions.run(
      outputs = [ctx.outputs.out],
      executable = 'dummy')
  files_to_build = depset([ctx.outputs.out])
  return DefaultInfo(
      files = files_to_build,
  )

test_rule = rule(
    implementation=_test_impl,
    attrs = {
        "out": attr.output(mandatory = True),
    },
)
EOF

  bazel aquery --experimental_allow_tags_propagation '//test:test' > output1 2> $TEST_log \
      || fail "should have generated output successfully"

  assert_contains "ExecutionInfo: {local: '', no-cache: '', no-remote: '', no-sandbox: '', requires-network: ''}" output1
}

# Test a basic Starlark ctx.actions.run rule which has tags, that should be
# propagated, when the rule also has execution_info.
function test_tags_propagated_to_run_with_exec_info_in_rule() {
  mkdir -p test
  cat << EOF >> test/BUILD
load(":starlark.bzl", "test_rule")

test_rule(
    name = "test",
    out = "output.txt",
    tags = ["no-cache", "no-remote", "custom-tag-1", "requires-network", "local"]
)
EOF

  cat << 'EOF' >> test/starlark.bzl
def _test_impl(ctx):
  ctx.actions.run(
      outputs = [ctx.outputs.out],
      executable = 'dummy',
      execution_requirements = {"requires-x": "", "custom-tag-whatever": "", "no-cache": "1"})
  files_to_build = depset([ctx.outputs.out])
  return DefaultInfo(
      files = files_to_build,
  )

test_rule = rule(
    implementation=_test_impl,
    attrs = {
        "out": attr.output(mandatory = True),
    },
)
EOF

  bazel aquery --experimental_allow_tags_propagation '//test:test' > output1 2> $TEST_log \
      || fail "should have generated output successfully"

  assert_contains "ExecutionInfo: {local: '', no-cache: 1, no-remote: '', requires-network: '', requires-x: ''}" output1
}

# Test a basic Starlark ctx.actions.run rule which has tags, that should not be
# propagated as --experimental_allow_tags_propagation flag set to false.
function test_tags_not_propagated_to_run_when_incompatible_flag_off() {
  mkdir -p test
  cat << EOF >> test/BUILD
load(":starlark.bzl", "test_rule")

test_rule(
    name = "test",
    out = "output.txt",
    tags = ["no-cache", "no-remote", "no-sandbox", "requires-network", "local"]
)
EOF

  cat << 'EOF' >> test/starlark.bzl
def _test_impl(ctx):
  ctx.actions.run(
      outputs = [ctx.outputs.out],
      executable = 'dummy')
  files_to_build = depset([ctx.outputs.out])
  return DefaultInfo(
      files = files_to_build,
  )

test_rule = rule(
    implementation=_test_impl,
    attrs = {
        "out": attr.output(mandatory = True),
    },
)
EOF

  bazel aquery --experimental_allow_tags_propagation=false '//test:test' > output1 2> $TEST_log \
      || fail "should have generated output successfully"

  assert_not_contains "ExecutionInfo: {}" output1
}

run_suite "tags propagation: Starlark rule tests"
