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
  toolchain_name=${1:-test_toolchain}
  mkdir -p toolchain
  cat >> toolchain/toolchain_${toolchain_name}.bzl <<EOF
def _impl(ctx):
  toolchain = platform_common.ToolchainInfo(
      extra_label = ctx.attr.extra_label,
      extra_str = ctx.attr.extra_str)
  return [toolchain]

${toolchain_name} = rule(
    implementation = _impl,
    attrs = {
        'extra_label': attr.label(),
        'extra_str': attr.string(),
    }
)
EOF

  cat >> toolchain/BUILD <<EOF
toolchain_type(name = '${toolchain_name}',
    visibility = ['//visibility:public'])
EOF
}

function write_test_rule() {
  rule_name=${1:-use_toolchain}
  toolchain_name=${2:-test_toolchain}
  mkdir -p toolchain
  cat >> toolchain/rule_${rule_name}.bzl <<EOF
def _impl(ctx):
  toolchain = ctx.toolchains['//toolchain:${toolchain_name}']
  message = ctx.attr.message
  print(
      'Using toolchain: rule message: "%s", toolchain extra_str: "%s"' %
         (message, toolchain.extra_str))
  return []

${rule_name} = rule(
    implementation = _impl,
    attrs = {
        'message': attr.string(),
    },
    toolchains = ['//toolchain:${toolchain_name}'],
)
EOF
}

function write_test_aspect() {
  aspect_name=${1:-use_toolchain}
  toolchain_name=${2:-test_toolchain}
  mkdir -p toolchain
  cat >> toolchain/aspect_${aspect_name}.bzl <<EOF
def _impl(target, ctx):
  toolchain = ctx.toolchains['//toolchain:${toolchain_name}']
  message = ctx.rule.attr.message
  print(
      'Using toolchain in aspect: rule message: "%s", toolchain extra_str: "%s"' %
          (message, toolchain.extra_str))
  return []

${aspect_name} = aspect(
    implementation = _impl,
    attrs = {},
    toolchains = ['//toolchain:${toolchain_name}'],
)
EOF
}

function write_register_toolchain() {
  toolchain_name=${1:-test_toolchain}
  cat >> WORKSPACE <<EOF
register_toolchains('//:${toolchain_name}_1')
EOF

  cat >> BUILD <<EOF
load('//toolchain:toolchain_${toolchain_name}.bzl', '${toolchain_name}')

# Define the toolchain.
filegroup(name = 'dep_rule_${toolchain_name}')
${toolchain_name}(
    name = '${toolchain_name}_impl_1',
    extra_label = ':dep_rule_${toolchain_name}',
    extra_str = 'foo from ${toolchain_name}',
    visibility = ['//visibility:public'])

# Declare the toolchain.
toolchain(
    name = '${toolchain_name}_1',
    toolchain_type = '//toolchain:${toolchain_name}',
    exec_compatible_with = [],
    target_compatible_with = [],
    toolchain = ':${toolchain_name}_impl_1',
    visibility = ['//visibility:public'])
EOF
}

function test_toolchain_provider() {
  write_test_toolchain

  cat >> BUILD <<EOF
load('//toolchain:toolchain_test_toolchain.bzl', 'test_toolchain')
load('//report:report.bzl', 'report_toolchain')

filegroup(name = 'dep_rule')
test_toolchain(
    name = 'linux_toolchain',
    extra_label = ':dep_rule',
    extra_str = 'bar',
)
report_toolchain(
  name = 'report',
  fields = ['extra_label', 'extra_str'],
  toolchain = ':linux_toolchain',
)
EOF

  bazel build //:report &> $TEST_log || fail "Build failed"
  expect_log 'extra_label = "//:dep_rule"'
  expect_log 'extra_str = "bar"'
}

function test_toolchain_use_in_rule {
  write_test_toolchain
  write_test_rule
  write_register_toolchain

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain: rule message: "this is the rule", toolchain extra_str: "foo from test_toolchain"'
}

function test_toolchain_alias_use_in_rule {
  write_test_toolchain
  write_test_rule

  cat >> BUILD <<EOF
load('//toolchain:toolchain_test_toolchain.bzl', 'test_toolchain')

# Define the toolchain.
filegroup(name = 'dep_rule_test_toolchain')
test_toolchain(
    name = 'test_toolchain_impl_1',
    extra_label = ':dep_rule_test_toolchain',
    extra_str = 'foo from test_toolchain',
    visibility = ['//visibility:public'])
alias(
    name = 'test_toolchain_impl_1_alias',
    actual = ':test_toolchain_impl_1',
    visibility = ['//visibility:public'])

# Declare the toolchain.
toolchain(
    name = 'test_toolchain_1',
    toolchain_type = '//toolchain:test_toolchain',
    exec_compatible_with = [],
    target_compatible_with = [],
    toolchain = ':test_toolchain_impl_1_alias',
    visibility = ['//visibility:public'])
EOF

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build --extra_toolchains=//:test_toolchain_1 //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain: rule message: "this is the rule", toolchain extra_str: "foo from test_toolchain"'
}

function test_toolchain_alias_chain_use_in_rule {
  write_test_toolchain
  write_test_rule

  cat >> BUILD <<EOF
load('//toolchain:toolchain_test_toolchain.bzl', 'test_toolchain')

# Define the toolchain.
filegroup(name = 'dep_rule_test_toolchain')
test_toolchain(
    name = 'test_toolchain_impl_1',
    extra_label = ':dep_rule_test_toolchain',
    extra_str = 'foo from test_toolchain',
    visibility = ['//visibility:public'])
alias(
    name = 'test_toolchain_impl_1_alias_alpha',
    actual = ':test_toolchain_impl_1',
    visibility = ['//visibility:public'])
alias(
    name = 'test_toolchain_impl_1_alias_beta',
    actual = ':test_toolchain_impl_1_alias_alpha',
    visibility = ['//visibility:public'])

# Declare the toolchain.
toolchain(
    name = 'test_toolchain_1',
    toolchain_type = '//toolchain:test_toolchain',
    exec_compatible_with = [],
    target_compatible_with = [],
    toolchain = ':test_toolchain_impl_1_alias_beta',
    visibility = ['//visibility:public'])
EOF

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build --extra_toolchains=//:test_toolchain_1 //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain: rule message: "this is the rule", toolchain extra_str: "foo from test_toolchain"'
}

function test_toolchain_use_in_rule_missing {
  write_test_toolchain
  write_test_rule
  #rite_register_toolchain
  # Do not register test_toolchain to trigger the error.

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log 'While resolving toolchains for target //demo:use: no matching toolchains found for types //toolchain:test_toolchain'
}

function test_multiple_toolchain_use_in_rule {
  write_test_toolchain test_toolchain_1
  write_test_toolchain test_toolchain_2

  write_register_toolchain test_toolchain_1
  write_register_toolchain test_toolchain_2

  # The rule uses two separate toolchains.
  mkdir -p toolchain
  cat >> toolchain/rule_use_toolchains.bzl <<EOF
def _impl(ctx):
  toolchain_1 = ctx.toolchains['//toolchain:test_toolchain_1']
  toolchain_2 = ctx.toolchains['//toolchain:test_toolchain_2']
  message = ctx.attr.message
  print(
      'Using toolchain: rule message: "%s", toolchain 1 extra_str: "%s", toolchain 2 extra_str: "%s"' %
         (message, toolchain_1.extra_str, toolchain_2.extra_str))
  return []

use_toolchains = rule(
    implementation = _impl,
    attrs = {
        'message': attr.string(),
    },
    toolchains = [
        '//toolchain:test_toolchain_1',
        '//toolchain:test_toolchain_2',
    ],
)
EOF

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchains.bzl', 'use_toolchains')
# Use the toolchain.
use_toolchains(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain: rule message: "this is the rule", toolchain 1 extra_str: "foo from test_toolchain_1", toolchain 2 extra_str: "foo from test_toolchain_2"'
}

function test_multiple_toolchain_use_in_rule_one_missing {
  write_test_toolchain test_toolchain_1
  write_test_toolchain test_toolchain_2

  write_register_toolchain test_toolchain_1
  # Do not register test_toolchain_2 to cause the error,

  # The rule uses two separate toolchains.
  mkdir -p toolchain
  cat >> toolchain/rule_use_toolchains.bzl <<EOF
def _impl(ctx):
  toolchain_1 = ctx.toolchains['//toolchain:test_toolchain_1']
  toolchain_2 = ctx.toolchains['//toolchain:test_toolchain_2']
  message = ctx.attr.message
  print(
      'Using toolchain: rule message: "%s", toolchain 1 extra_str: "%s", toolchain 2 extra_str: "%s"' %
         (message, toolchain_1.extra_str, toolchain_2.extra_str))
  return []

use_toolchains = rule(
    implementation = _impl,
    attrs = {
        'message': attr.string(),
    },
    toolchains = [
        '//toolchain:test_toolchain_1',
        '//toolchain:test_toolchain_2',
    ],
)
EOF

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchains.bzl', 'use_toolchains')
# Use the toolchain.
use_toolchains(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log 'While resolving toolchains for target //demo:use: no matching toolchains found for types //toolchain:test_toolchain_2'
}

function test_toolchain_use_in_rule_non_required_toolchain {
  write_test_toolchain
  write_register_toolchain

  # The rule argument toolchains requires one toolchain, but the implementation requests a different
  # one.
  mkdir -p toolchain
  cat >> toolchain/rule_use_toolchain.bzl <<EOF
def _impl(ctx):
  toolchain = ctx.toolchains['//toolchain:wrong_toolchain']
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

  # Trigger the wrong toolchain.
  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log 'In use_toolchain rule //demo:use, toolchain type //toolchain:wrong_toolchain was requested but only types \[//toolchain:test_toolchain\] are configured'
}

function test_toolchain_debug_messages {
  write_test_toolchain
  write_test_rule
  write_register_toolchain

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build \
    --toolchain_resolution_debug \
    --incompatible_auto_configure_host_platform \
    //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'ToolchainResolution: Looking for toolchain of type //toolchain:test_toolchain'
  expect_log 'ToolchainResolution:   For toolchain type //toolchain:test_toolchain, possible execution platforms and toolchains: {@local_config_platform//:host -> //:test_toolchain_impl_1}'
  expect_log 'ToolchainResolution: Selected execution platform @local_config_platform//:host, type //toolchain:test_toolchain -> toolchain //:test_toolchain_impl_1'
  expect_log 'Using toolchain: rule message: "this is the rule", toolchain extra_str: "foo from test_toolchain"'
}

function test_toolchain_use_in_aspect {
  write_test_toolchain
  write_test_aspect
  write_register_toolchain

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
    --aspects //toolchain:aspect_use_toolchain.bzl%use_toolchain \
    //demo:demo &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain in aspect: rule message: "bar from demo", toolchain extra_str: "foo from test_toolchain"'
}

function test_toolchain_use_in_aspect_non_required_toolchain {
  write_test_toolchain
  write_register_toolchain

  # The aspect argument toolchains requires one toolchain, but the implementation requests a
  # different one.
  mkdir -p toolchain
  cat >> toolchain/aspect_use_toolchain.bzl <<EOF
def _impl(target, ctx):
  toolchain = ctx.toolchains['//toolchain:wrong_toolchain']
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

  # Trigger the wrong toolchain.
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
    --aspects //toolchain:aspect_use_toolchain.bzl%use_toolchain \
    //demo:demo &> $TEST_log && fail "Build failure expected"
  expect_log 'In aspect //toolchain:aspect_use_toolchain.bzl%use_toolchain applied to demo rule //demo:demo, toolchain type //toolchain:wrong_toolchain was requested but only types \[//toolchain:test_toolchain\] are configured'
}

function test_toolchain_constraints() {
  write_test_toolchain
  write_test_rule

  cat >> WORKSPACE <<EOF
register_toolchains('//:toolchain_1')
register_toolchains('//:toolchain_2')
EOF

  cat >> BUILD <<EOF
load('//toolchain:toolchain_test_toolchain.bzl', 'test_toolchain')

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
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  # This should use toolchain_1.
  bazel build \
    --host_platform=//:platform1 \
    --platforms=//:platform2 \
    //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain: rule message: "this is the rule", toolchain extra_str: "foo from 1"'

  # This should use toolchain_2.
  bazel build \
    --host_platform=//:platform2 \
    --platforms=//:platform1 \
    //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain: rule message: "this is the rule", toolchain extra_str: "foo from 2"'

  # This should not match any toolchains.
  bazel build \
    --host_platform=//:platform1 \
    --platforms=//:platform1 \
    //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log 'While resolving toolchains for target //demo:use: no matching toolchains found for types //toolchain:test_toolchain'
  expect_not_log 'Using toolchain: rule message:'
}

function test_register_toolchain_error_invalid_label() {
  write_test_toolchain
  write_test_rule
  write_register_toolchain

  cat >> WORKSPACE <<EOF
register_toolchains('/:invalid:label:syntax')
EOF

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log "invalid registered toolchain '/:invalid:label:syntax': not a valid absolute pattern"
}

function test_register_toolchain_error_invalid_target() {
  write_test_toolchain
  write_test_rule
  write_register_toolchain

  cat > WORKSPACE <<EOF
register_toolchains('//demo:not_a_target')
EOF

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log "While resolving toolchains for target //demo:use: invalid registered toolchain '//demo:not_a_target': no such target '//demo:not_a_target': target 'not_a_target' not declared in package 'demo'"
}

function test_register_toolchain_error_target_not_a_toolchain() {
  write_test_toolchain
  write_test_rule
  write_register_toolchain

  cat >> WORKSPACE <<EOF
register_toolchains('//demo:invalid')
EOF

  mkdir -p demo
  cat >> demo/out.log<<EOF
INVALID
EOF
  cat >> demo/BUILD <<EOF
filegroup(
    name = "invalid",
    srcs = ["out.log"],
)

load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log "While resolving toolchains for target //demo:use: invalid registered toolchain '//demo:invalid': target does not provide the DeclaredToolchainInfo provider"
}


function test_register_toolchain_error_invalid_pattern() {
  cat >WORKSPACE <<EOF
register_toolchains('//:bad1')
register_toolchains('//:bad2')
EOF

  cat >rules.bzl <<EOF
def _impl(ctx):
  toolchain = ctx.toolchains['//:dummy']
  return []

foo = rule(
  implementation = _impl,
  toolchains = ['//:dummy'],
)
EOF

  cat >BUILD <<EOF
load(":rules.bzl", "foo")
toolchain_type(name = 'dummy')
foo(name = "foo")
EOF

  bazel build //:foo &> $TEST_log && fail "Build failure expected"
  # It's uncertain which error will happen first, so handle either.
  expect_log "While resolving toolchains for target //:foo: invalid registered toolchain '//:bad[12]': no such target"
}


function test_toolchain_error_invalid_target() {
  write_test_toolchain
  write_test_rule

  # Write toolchain with an invalid target.
  mkdir -p invalid
  cat > invalid/BUILD <<EOF
toolchain(
    name = 'invalid_toolchain',
    toolchain_type = '//toolchain:test_toolchain',
    exec_compatible_with = [],
    target_compatible_with = [],
    toolchain = '//toolchain:does_not_exist',
    visibility = ['//visibility:public'])
EOF

  cat > WORKSPACE <<EOF
register_toolchains('//invalid:invalid_toolchain')
EOF

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log "Target '//demo:use' depends on toolchain '//toolchain:does_not_exist', which cannot be found: no such target '//toolchain:does_not_exist': target 'does_not_exist' not declared in package 'toolchain'"
}


function test_platforms_options_error_invalid_target() {
  write_test_toolchain
  write_test_rule
  write_register_toolchain

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  # Write an invalid rule to be the platform.
  mkdir -p platform
  cat >> platform/BUILD <<EOF
filegroup(name = 'not_a_platform')
EOF

  bazel build --platforms=//platform:not_a_platform //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log "While resolving toolchains for target //demo:use: Target //platform:not_a_platform was referenced as a platform, but does not provide PlatformInfo"

  bazel build --host_platform=//platform:not_a_platform //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log "While resolving toolchains for target //demo:use: Target //platform:not_a_platform was referenced as a platform, but does not provide PlatformInfo"
}


function test_native_rule_target_exec_constraints() {
  mkdir -p platform
  cat >> platform/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
constraint_setting(name = "test")

constraint_value(
    name = "test_enabled",
    constraint_setting = ":test",
)

platform(
    name = "test_platform",
    constraint_values = [
        ":test_enabled",
    ],
)
EOF

  mkdir -p demo
  cat >> demo/BUILD <<EOF
genrule(
    name = "target",
    outs = ["out.txt"],
    cmd = """
      echo "platform" > \$@
    """,
    exec_compatible_with = [
        "//platform:test_enabled",
    ],
)
EOF

  # When no platform has the constraint, an error
  bazel build \
    --toolchain_resolution_debug \
    //demo:target &> $TEST_log && fail "Build failure expected"
  expect_log "While resolving toolchains for target //demo:target: .* from available execution platforms \[\]"

  # When the platform exists, it is used.
  bazel build \
    --extra_execution_platforms=//platform:test_platform \
    --toolchain_resolution_debug \
    //demo:target &> $TEST_log || fail "Build failed"
  expect_log "ToolchainResolution: Selected execution platform //platform:test_platform"
}


function test_rule_with_default_execution_constraints() {
  write_test_toolchain
  write_register_toolchain

  # Add test platforms.
  mkdir -p platforms
  cat >> platforms/BUILD <<EOF
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
EOF

  # Add a rule with default execution constraints.
  mkdir -p demo
  cat >> demo/rule.bzl <<EOF
def _impl(ctx):
  return []

sample_rule = rule(
  implementation = _impl,
  attrs = {},
  exec_compatible_with = [
    '//platforms:value2',
  ],
  toolchains = ['//toolchain:test_toolchain'],
)
EOF

  # Use the new rule.
  cat >> demo/BUILD <<EOF
load(':rule.bzl', 'sample_rule')

sample_rule(name = 'use')
EOF

  # Build the target, using debug messages to verify the correct platform was selected.
  bazel build \
    --extra_execution_platforms=//platforms:all \
    --toolchain_resolution_debug \
    //demo:use &> $TEST_log || fail "Build failed"
    expect_log "Selected execution platform //platforms:platform2"
}


function test_target_with_execution_constraints() {
  write_test_toolchain
  write_register_toolchain

  # Add test platforms.
  mkdir -p platforms
  cat >> platforms/BUILD <<EOF
package(default_visibility = ['//visibility:public'])
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
EOF

  # Add a rule with default execution constraints.
  mkdir -p demo
  cat >> demo/rule.bzl <<EOF
def _impl(ctx):
  return []

sample_rule = rule(
  implementation = _impl,
  attrs = {},
  toolchains = ['//toolchain:test_toolchain'],
  execution_platform_constraints_allowed = True,
)
EOF

  # Use the new rule.
  cat >> demo/BUILD <<EOF
load(':rule.bzl', 'sample_rule')

sample_rule(
  name = 'use',
  exec_compatible_with = [
    '//platforms:value2',
  ],
)
EOF

  # Build the target, using debug messages to verify the correct platform was selected.
  bazel build \
    --extra_execution_platforms=//platforms:all \
    --toolchain_resolution_debug \
    //demo:use &> $TEST_log || fail "Build failed"
    expect_log "Selected execution platform //platforms:platform2"
}

function test_rule_and_target_with_execution_constraints() {
  write_test_toolchain
  write_register_toolchain

  # Add test platforms.
  mkdir -p platforms
  cat >> platforms/BUILD <<EOF
package(default_visibility = ['//visibility:public'])
constraint_setting(name = 'setting1')
constraint_value(name = 'value1', constraint_setting = ':setting1')
constraint_value(name = 'value2', constraint_setting = ':setting1')

constraint_setting(name = 'setting2')
constraint_value(name = 'value3', constraint_setting = ':setting2')
constraint_value(name = 'value4', constraint_setting = ':setting2')

platform(
    name = 'platform1_3',
    constraint_values = [':value1', ':value3'],
    visibility = ['//visibility:public'])
platform(
    name = 'platform1_4',
    constraint_values = [':value1', ':value4'],
    visibility = ['//visibility:public'])
platform(
    name = 'platform2_3',
    constraint_values = [':value2', ':value3'],
    visibility = ['//visibility:public'])
platform(
    name = 'platform2_4',
    constraint_values = [':value2', ':value4'],
    visibility = ['//visibility:public'])
EOF

  # Add a rule with default execution constraints.
  mkdir -p demo
  cat >> demo/rule.bzl <<EOF
def _impl(ctx):
  return []

sample_rule = rule(
  implementation = _impl,
  attrs = {},
  exec_compatible_with = [
    '//platforms:value2',
  ],
  toolchains = ['//toolchain:test_toolchain'],
  execution_platform_constraints_allowed = True,
)
EOF

  # Use the new rule.
  cat >> demo/BUILD <<EOF
load(':rule.bzl', 'sample_rule')

sample_rule(
  name = 'use',
  exec_compatible_with = [
    '//platforms:value4',
  ],
)
EOF

  # Build the target, using debug messages to verify the correct platform was selected.
  bazel build \
    --extra_execution_platforms=//platforms:all \
    --toolchain_resolution_debug \
    //demo:use &> $TEST_log || fail "Build failed"
    expect_log "Selected execution platform //platforms:platform2_4"
}

function test_default_constraint_values {
  # Add test constraints and platforms.
  mkdir -p platforms
  cat >> platforms/BUILD <<EOF
package(default_visibility = ['//visibility:public'])
constraint_setting(name = 'setting1', default_constraint_value = ':value_foo')
constraint_value(name = 'value_foo', constraint_setting = ':setting1')
constraint_value(name = 'value_bar', constraint_setting = ':setting1')

platform(
    name = 'platform_default',
    constraint_values = [])
platform(
    name = 'platform_no_default',
    constraint_values = [':value_bar'])
EOF

  # Add test toolchains using the constraints.
  write_test_toolchain
  cat >> BUILD <<EOF
load('//toolchain:toolchain_test_toolchain.bzl', 'test_toolchain')

# Define the toolchains.
test_toolchain(
    name = 'test_toolchain_impl_foo',
    extra_str = 'foo',
    visibility = ['//visibility:public'])

test_toolchain(
    name = 'test_toolchain_impl_bar',
    extra_str = 'bar',
    visibility = ['//visibility:public'])

# Declare the toolchains.
toolchain(
    name = 'test_toolchain_foo',
    toolchain_type = '//toolchain:test_toolchain',
    exec_compatible_with = [],
    target_compatible_with = [
      # No constraint set, takes the default.
    ],
    toolchain = ':test_toolchain_impl_foo',
    visibility = ['//visibility:public'])
toolchain(
    name = 'test_toolchain_bar',
    toolchain_type = '//toolchain:test_toolchain',
    exec_compatible_with = [],
    target_compatible_with = [
      # Explicitly sets a non-default value.
      '//platforms:value_bar',
    ],
    toolchain = ':test_toolchain_impl_bar',
    visibility = ['//visibility:public'])
EOF

  # Register the toolchains
  cat >> WORKSPACE <<EOF
register_toolchains('//:test_toolchain_foo', '//:test_toolchain_bar')
EOF

  write_test_rule
  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  # Test some builds and verify which was used.
  # This should use the default value.
  bazel build \
    --platforms=//platforms:platform_default \
    //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'toolchain extra_str: "foo"'

  # This should use the explicit value.
  bazel build \
    --platforms=//platforms:platform_no_default \
    //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'toolchain extra_str: "bar"'
}

function test_make_variables_custom_rule() {
  # Create a toolchain rule that also exposes make variables.
  mkdir -p toolchain
  cat >> toolchain/BUILD <<EOF
toolchain_type(name = 'toolchain_var',
    visibility = ['//visibility:public'])
EOF
  cat >> toolchain/toolchain_var.bzl <<EOF
def _impl(ctx):
  toolchain = platform_common.ToolchainInfo()
  value = ctx.attr.value
  templates = platform_common.TemplateVariableInfo({'VALUE': value})
  return [toolchain, templates]

toolchain_var = rule(
    implementation = _impl,
    attrs = {
        'value': attr.string(mandatory = True),
    }
)
EOF

  # Create a rule that consumes the toolchain.
  cat >> toolchain/rule_var.bzl <<EOF
def _impl(ctx):
  toolchain = ctx.toolchains['//toolchain:toolchain_var']
  value = ctx.var['VALUE']
  print('Using toolchain: value "%s"' % value)
  return []

rule_var = rule(
    implementation = _impl,
    toolchains = ['//toolchain:toolchain_var'],
)
EOF

  # Create and register a toolchain
  cat >> WORKSPACE <<EOF
register_toolchains('//:toolchain_var_1')
EOF

  cat >> BUILD <<EOF
load('//toolchain:toolchain_var.bzl', 'toolchain_var')

# Define the toolchain.
toolchain_var(
    name = 'toolchain_var_impl_1',
    value = 'foo',
    visibility = ['//visibility:public'])

# Declare the toolchain.
toolchain(
    name = 'toolchain_var_1',
    toolchain_type = '//toolchain:toolchain_var',
    exec_compatible_with = [],
    target_compatible_with = [],
    toolchain = ':toolchain_var_impl_1',
    visibility = ['//visibility:public'])
EOF

  # Instantiate the rule and verify the output.
  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_var.bzl', 'rule_var')
rule_var(name = 'demo')
EOF

  bazel build //demo:demo &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain: value "foo"'
}

function test_local_config_platform() {
  bazel query @local_config_platform//... &> $TEST_log || fail "Build failed"
  expect_log '@local_config_platform//:host'
}

# Test cycles in registered toolchains, which can only happen when
# registered_toolchains is called for something that is not actually
# using the "toolchain" rule.
function test_registered_toolchain_cycle() {

  # Set up two sets of rules and toolchains, one depending on the other.
  cat >>lower.bzl <<EOF
def _lower_toolchain_impl(ctx):
  message = ctx.attr.message
  toolchain = platform_common.ToolchainInfo(
      message=message)
  return [toolchain]

lower_toolchain = rule(
    implementation = _lower_toolchain_impl,
    attrs = {
        'message': attr.string(),
    },
)

def _lower_library_impl(ctx):
  toolchain = ctx.toolchains['//:lower']
  print('lower library: %s' % toolchain.message)
  return []

lower_library = rule(
    implementation = _lower_library_impl,
    attrs = {},
    toolchains = ['//:lower'],
)
EOF
  cat >>upper.bzl <<EOF
def _upper_toolchain_impl(ctx):
  tool_message = ctx.toolchains['//:lower'].message
  message = ctx.attr.message
  toolchain = platform_common.ToolchainInfo(
      tool_message=tool_message,
      message=message)
  return [toolchain]

upper_toolchain = rule(
    implementation = _upper_toolchain_impl,
    attrs = {
        'message': attr.string(),
    },
    toolchains = ['//:lower'],
)

def _upper_library_impl(ctx):
  toolchain = ctx.toolchains['//:upper']
  print('upper library: %s (%s)' % (toolchain.message, toolchain.tool_message))
  return []

upper_library = rule(
    implementation = _upper_library_impl,
    attrs = {},
    toolchains = ['//:upper'],
)
EOF

  # Define the actual targets using these.
  cat >>BUILD <<EOF
load('//:lower.bzl', 'lower_toolchain', 'lower_library')
load('//:upper.bzl', 'upper_toolchain', 'upper_library')

toolchain_type(name = 'lower')
toolchain_type(name = 'upper')

lower_library(
    name = 'lower_lib',
)

lower_toolchain(
    name = 'lower_toolchain',
    message = 'hi from lower',
)
toolchain(
    name = 'lower_toolchain_impl',
    toolchain_type = '//:lower',
    toolchain = ':lower_toolchain',
)

upper_library(
    name = 'upper_lib',
)

upper_toolchain(
    name = 'upper_toolchain',
    message = 'hi from upper',
)
toolchain(
    name = 'upper_toolchain_impl',
    toolchain_type = '//:upper',
    toolchain = ':upper_toolchain',
)
EOF

  # Finally, set up the misconfigured WORKSPACE file.
  cat >>WORKSPACE <<EOF
register_toolchains(
    '//:upper_toolchain', # Not a toolchain() target!
    '//:lower_toolchain_impl',
    )
EOF

  # Execute the build and check the error message.
  bazel build //:upper_lib &> $TEST_log && fail "Build succeeded unexpectedly"
  expect_not_log "java.lang.IllegalStateException"
  expect_log "Misconfigured toolchains: //:upper_toolchain is declared as a toolchain but has inappropriate dependencies"
}

function test_platform_duplicate_constraint_error() {
  # Write a platform with duplicate constraint values for the same setting.
  mkdir -p platform
  cat >> platform/BUILD <<EOF
constraint_setting(name = 'foo')
constraint_value(name = 'val1', constraint_setting = ':foo')
constraint_value(name = 'val2', constraint_setting = ':foo')
platform(
    name = 'test',
    constraint_values = [
        ':val1',
        ':val2',
    ],
)
EOF

  bazel build //platform:test &> $TEST_log && fail "Build failure expected"
  expect_log "Duplicate constraint values detected"
}

function test_toolchain_duplicate_constraint_error() {
  # Write a toolchain with duplicate constraint values for the same setting.
  mkdir -p toolchain
  cat >> toolchain/BUILD <<EOF
constraint_setting(name = 'foo')
constraint_value(name = 'val1', constraint_setting = ':foo')
constraint_value(name = 'val2', constraint_setting = ':foo')
constraint_setting(name = 'bar')
constraint_value(name = 'val3', constraint_setting = ':bar')
constraint_value(name = 'val4', constraint_setting = ':bar')
toolchain_type(name = 'toolchain_type')
filegroup(name = 'toolchain')
toolchain(
    name = 'test',
    toolchain_type = ':toolchain_type',
    exec_compatible_with = [
        ':val1',
        ':val2',
    ],
    target_compatible_with = [
        ':val3',
        ':val4',
    ],
    toolchain = ':toolchain',
)
EOF

  bazel build //toolchain:test &> $TEST_log && fail "Build failure expected"
  expect_not_log "java.lang.IllegalArgumentException"
  expect_log "in exec_compatible_with attribute of toolchain rule //toolchain:test: Duplicate constraint values detected: constraint_setting //toolchain:foo has \[//toolchain:val1, //toolchain:val2\]"
  expect_log "in target_compatible_with attribute of toolchain rule //toolchain:test: Duplicate constraint values detected: constraint_setting //toolchain:bar has \[//toolchain:val3, //toolchain:val4\]"
}

# TODO(katre): Test using toolchain-provided make variables from a genrule.

run_suite "toolchain tests"
