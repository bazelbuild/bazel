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

  # Create shared constraints.
  mkdir -p constraints
  cat >>constraints/BUILD <<EOF
package(default_visibility = ['//visibility:public'])
constraint_setting(name = 'os')
constraint_value(name = 'linux',
    constraint_setting = ':os')
constraint_value(name = 'mac',
    constraint_setting = ':os')
EOF

  # Create shared report rule for printing info.
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
  _report_impl,
  attrs = {
    'fields': attr.string_list(),
    'toolchain': attr.label(providers = [platform_common.ToolchainInfo]),
  }
)
EOF
}

function test_toolchain_rule() {

  mkdir -p toolchain
  cat >> toolchain/toolchain.bzl <<EOF
def _test_toolchain_impl(ctx):
  toolchain = platform_common.toolchain(
      exec_compatible_with = ctx.attr.exec_compatible_with,
      target_compatible_with = ctx.attr.target_compatible_with,
      extra_label = ctx.attr.extra_label,
      extra_str = ctx.attr.extra_str)
  return [toolchain]

test_toolchain = rule(
    _test_toolchain_impl,
    attrs = {
        'exec_compatible_with': attr.label_list(providers = [platform_common.ConstraintValueInfo]),
        'target_compatible_with': attr.label_list(providers = [platform_common.ConstraintValueInfo]),
       'extra_label': attr.label(),
       'extra_str': attr.string(),
    }
)
EOF

  cat >> toolchain/BUILD <<EOF
load('//report:report.bzl', 'report_toolchain')
load(':toolchain.bzl', 'test_toolchain')
filegroup(name = 'dep_rule')
test_toolchain(
    name = 'linux_toolchain',
    exec_compatible_with = [
      '//constraints:linux',
    ],
    target_compatible_with = [
      '//constraints:mac',
    ],
    extra_label = ':dep_rule',
    extra_str = 'bar',
)
report_toolchain(
  name = 'report',
  fields = ['extra_label', 'extra_str'],
  toolchain = ':linux_toolchain',
)
EOF

  bazel build //toolchain:report &> $TEST_log || fail "Build failed"
  expect_log 'extra_label = "//toolchain:dep_rule"'
  expect_log 'extra_str = "bar"'
}

run_suite "toolchain tests"
