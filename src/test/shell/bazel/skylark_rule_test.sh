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
# Tests building with rules defined in Skylark.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# Test a basic skylark rule which touches an output file
function test_basic_output() {
  mkdir -p test
  cat << EOF >> test/BUILD
load(":skylark.bzl", "test_rule")

test_rule(
    name = "test",
    out = "output.txt"
)
EOF

  cat << 'EOF' >> test/skylark.bzl
def _test_impl(ctx):
  ctx.actions.run_shell(outputs = [ctx.outputs.out],
                        command = ["touch", ctx.outputs.out.path])
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

  bazel build //test:test &> $TEST_log \
      || fail "should have generated output successfully"
}

# Test a basic skylark rule which is valid except the action fails on execution.
function test_execution_failure() {
  mkdir -p test
  cat << EOF >> test/BUILD
load(":skylark.bzl", "test_rule")

test_rule(
    name = "test",
    out = "output.txt"
)
EOF

  cat << 'EOF' >> test/skylark.bzl
def _test_impl(ctx):
  ctx.actions.run_shell(outputs = [ctx.outputs.out],
                        command = ["not_a_command", ctx.outputs.out.path])
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

  ! bazel build //test:test &> $TEST_log \
      || fail "Should have resulted in an execution error"

  expect_log "error executing shell command"
}

run_suite "skylark rule definition tests"
