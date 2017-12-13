#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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
# Test rules with outputs definitions.
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }


function test_plain_outputs() {
  #create_new_workspace
  cat >rule.bzl <<EOF
def _impl(ctx):
  ctx.file_action(
      output=ctx.outputs.out,
      content="Hello World!"
  )
  return []

demo_rule = rule(
  _impl,
  attrs = {
    'foo': attr.string(),
  },
  outputs = {
    'out': '%{foo}.txt'
  })
EOF

  cat >BUILD <<EOF
load(':rule.bzl', 'demo_rule')

demo_rule(
  name = 'demo',
  foo = 'demo_output_name')
EOF

  bazel build //:demo &> $TEST_log || fail "Build failed"
  expect_log "demo_output_name.txt"
}

function test_function_outputs() {
  #create_new_workspace
  cat >rule.bzl <<EOF
def _outputs(foo):
  return {
    'out': foo + '.txt',
  }

def _impl(ctx):
  ctx.file_action(
      output=ctx.outputs.out,
      content="Hello World!"
  )
  return []

demo_rule = rule(
  _impl,
  attrs = {
    'foo': attr.string(),
  },
  outputs = _outputs)
EOF

  cat >BUILD <<EOF
load(':rule.bzl', 'demo_rule')

demo_rule(
  name = 'demo',
  foo = 'demo_output_name')
EOF

  bazel build //:demo &> $TEST_log || fail "Build failed"
  expect_log "demo_output_name.txt"
}

function test_output_select_error() {
  #create_new_workspace
  cat >rule.bzl <<EOF
def _impl(ctx):
  ctx.file_action(
      output=ctx.outputs.out,
      content="Hello World!"
  )
  return []

demo_rule = rule(
  _impl,
  attrs = {
    'foo': attr.string(),
  },
  outputs = select({
    '//conditions:default': {
      'out': '%{foo}.txt'
    }
  }))
EOF

  cat >BUILD <<EOF
load(':rule.bzl', 'demo_rule')

demo_rule(
  name = 'demo',
  foo = 'a_str')
EOF

  bazel build //:demo &> $TEST_log && fail "Build expected to fail"
  expect_log "expected dict or dict-returning function or NoneType for 'outputs' while calling rule but got select of dict instead"
}

function test_configurable_output_error() {
  #create_new_workspace
  cat >rule.bzl <<EOF
def _impl(ctx):
  ctx.file_action(
      output=ctx.outputs.out,
      content="Hello World!"
  )
  return []

demo_rule = rule(
  _impl,
  attrs = {
    'foo': attr.string(),
  },
  outputs = {
    'out': '%{foo}.txt'
  })
EOF

  cat >BUILD <<EOF
load(':rule.bzl', 'demo_rule')

demo_rule(
  name = 'demo',
  foo = select({
    '//conditions:default': 'selectable_str',
  }))
EOF

  bazel build //:demo &> $TEST_log && fail "Build expected to fail"
  expect_log "Attribute foo is configurable and cannot be used in outputs"
}

run_suite "skylark outputs tests"
