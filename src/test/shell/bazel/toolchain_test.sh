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
# Test the providers and rules related to toolchains.
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  create_new_workspace

  # Create shared report rule for printing toolchain info.
  mkdir report
  touch report/BUILD
  cat >>report/report.bzl <<EOF
def _report_impl(ctx):
  toolchain = ctx.attr.toolchain[platform_common.ToolchainInfo]
  for field in ctx.attr.fields:
    value = getattr(toolchain, field)
    if type(value) == 'Target':
      value = value.label
    print('%s = "%s"' % (field, value))

report_toolchain = rule(
    implementation = _report_impl,
    attrs = {
        'fields': attr.string_list(),
        'toolchain': attr.label(providers = [platform_common.ToolchainInfo]),
    }
)
EOF
}

function write_test_toolchain() {
  mkdir -p toolchain
  cat >> toolchain/toolchain.bzl <<EOF
def _test_toolchain_impl(ctx):
  toolchain = platform_common.ToolchainInfo(
      type = Label('//toolchain:test_toolchain'),
      extra_label = ctx.attr.extra_label,
      extra_str = ctx.attr.extra_str)
  return [toolchain]

test_toolchain = rule(
    implementation = _test_toolchain_impl,
    attrs = {
        'extra_label': attr.label(),
        'extra_str': attr.string(),
    }
)
EOF

  cat >> toolchain/BUILD <<EOF
toolchain_type(name = 'test_toolchain')
EOF
}

function write_test_rule() {
  mkdir -p toolchain
  cat >> toolchain/rule.bzl <<EOF
def _impl(ctx):
  toolchain = ctx.toolchains['//toolchain:test_toolchain']
  message = ctx.attr.message
  print(
      'Using toolchain: rule message: "%s", toolchain extra_str: "%s"' %
         (message, toolchain.extra_str))
  return []

use_toolchain = rule(
    implementation = _impl,
    attrs = {
        'message': attr.string(),
    },
    toolchains = ['//toolchain:test_toolchain'],
)
EOF
}

function write_test_aspect() {
  mkdir -p toolchain
  cat >> toolchain/aspect.bzl <<EOF
def _impl(target, ctx):
  toolchain = ctx.toolchains['//toolchain:test_toolchain']
  message = ctx.rule.attr.message
  print(
      'Using toolchain in aspect: rule message: "%s", toolchain extra_str: "%s"' %
          (message, toolchain.extra_str))
  return []

use_toolchain = aspect(
    implementation = _impl,
    attrs = {},
    toolchains = ['//toolchain:test_toolchain'],
)
EOF
}

function write_toolchains() {
  cat >> WORKSPACE <<EOF
register_toolchains('//:toolchain_1')
EOF

  cat >> BUILD <<EOF
load('//toolchain:toolchain.bzl', 'test_toolchain')

# Define the toolchain.
filegroup(name = 'dep_rule')
test_toolchain(
    name = 'toolchain_impl_1',
    extra_label = ':dep_rule',
    extra_str = 'foo from 1',
    visibility = ['//visibility:public'])

# Declare the toolchain.
toolchain(
    name = 'toolchain_1',
    toolchain_type = '//toolchain:test_toolchain',
    exec_compatible_with = [],
    target_compatible_with = [],
    toolchain = ':toolchain_impl_1',
    visibility = ['//visibility:public'])
EOF
}

function test_toolchain_provider() {
  write_test_toolchain

  cat >> BUILD <<EOF
load('//toolchain:toolchain.bzl', 'test_toolchain')
load('//report:report.bzl', 'report_toolchain')

filegroup(name = 'dep_rule')
test_toolchain(
    name = 'linux_toolchain',
    extra_label = ':dep_rule',
    extra_str = 'bar',
)
report_toolchain(
  name = 'report',
  fields = ['type', 'extra_label', 'extra_str'],
  toolchain = ':linux_toolchain',
)
EOF

  bazel build //:report &> $TEST_log || fail "Build failed"
  expect_log 'type = "//toolchain:test_toolchain"'
  expect_log 'extra_label = "//:dep_rule"'
  expect_log 'extra_str = "bar"'
}

function test_toolchain_use_in_rule {
  write_test_toolchain
  write_test_rule
  write_toolchains

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain: rule message: "this is the rule", toolchain extra_str: "foo from 1"'
}

function test_toolchain_use_in_aspect {
  write_test_toolchain
  write_test_aspect
  write_toolchains

  mkdir -p demo
  cat >> demo/demo.bzl <<EOF
def _impl(ctx):
  return []

demo = rule(
    implementation = _impl,
    attrs = {
        'message': attr.string(),
    }
)
EOF
  cat >> demo/BUILD <<EOF
load(':demo.bzl', 'demo')
demo(
    name = 'demo',
    message = 'bar from demo')
EOF

  bazel build \
    --aspects //toolchain:aspect.bzl%use_toolchain \
    //demo:demo &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain in aspect: rule message: "bar from demo", toolchain extra_str: "foo from 1"'
}

function test_toolchain_constraints() {
  write_test_toolchain
  write_test_rule

  cat >> WORKSPACE <<EOF
register_toolchains('//:toolchain_1')
register_toolchains('//:toolchain_2')
EOF

  cat >> BUILD <<EOF
load('//toolchain:toolchain.bzl', 'test_toolchain')

# Define constraints.
constraint_setting(name = 'setting')
constraint_value(name = 'value1', constraint_setting = ':setting')
constraint_value(name = 'value2', constraint_setting = ':setting')

platform(
    name = 'platform1',
    constraint_values = [':value1'],
    visibility = ['//visibility:public'])
platform(
    name = 'platform2',
    constraint_values = [':value2'],
    visibility = ['//visibility:public'])

# Define the toolchain.
filegroup(name = 'dep_rule')
test_toolchain(
    name = 'toolchain_impl_1',
    extra_label = ':dep_rule',
    extra_str = 'foo from 1',
    visibility = ['//visibility:public'])
test_toolchain(
    name = 'toolchain_impl_2',
    extra_label = ':dep_rule',
    extra_str = 'foo from 2',
    visibility = ['//visibility:public'])

# Declare the toolchain.
toolchain(
    name = 'toolchain_1',
    toolchain_type = '//toolchain:test_toolchain',
    exec_compatible_with = [':value1'],
    target_compatible_with = [':value2'],
    toolchain = ':toolchain_impl_1')
toolchain(
    name = 'toolchain_2',
    toolchain_type = '//toolchain:test_toolchain',
    exec_compatible_with = [':value2'],
    target_compatible_with = [':value1'],
    toolchain = ':toolchain_impl_2')
EOF

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  # This should use toolchain_1.
  bazel build \
    --experimental_host_platform=//:platform1 \
    --experimental_platforms=//:platform2 \
    //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain: rule message: "this is the rule", toolchain extra_str: "foo from 1"'

  # This should use toolchain_2.
  bazel build \
    --experimental_host_platform=//:platform2 \
    --experimental_platforms=//:platform1 \
    //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain: rule message: "this is the rule", toolchain extra_str: "foo from 2"'

  # This should not match any toolchains.
  bazel build \
    --experimental_host_platform=//:platform1 \
    --experimental_platforms=//:platform1 \
    //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log 'no matching toolchains found for types //toolchain:test_toolchain for target //demo:use'
  expect_not_log 'Using toolchain: rule message:'
}

run_suite "toolchain tests"
