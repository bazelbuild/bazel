#!/bin/bash
#
# Copyright 2020 The Bazel Authors. All rights reserved.
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
# Test the toolchain transition.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  create_new_workspace
}

function write_constraints() {
  mkdir constraint
  cat >constraint/BUILD.bazel <<EOF
package(default_visibility = ["//visibility:public"])

# Common constraints for testing.

constraint_setting(name = "type")

constraint_value(
    name = "target",
    constraint_setting = ":type",
)

constraint_value(
    name = "exec",
    constraint_setting = ":type",
)

constraint_value(
    name = "host",
    constraint_setting = ":type",
)

constraint_setting(name = "level")

constraint_value(
    name = "alpha",
    constraint_setting = ":level",
)

constraint_value(
    name = "beta",
    constraint_setting = ":level",
)
EOF
}

function write_platforms() {
  mkdir platform
  cat >platform/platform.bzl <<EOF
ShowPlatformInfo = provider(fields = ["platform_type", "level"])

def describe_platform_info(platform_info):
    if platform_info.level:
        return "%s-%s" % (platform_info.platform_type, platform_info.level)
    else:
        return platform_info.platform_type

def _show_platform_impl(ctx):
    target_constraint = ctx.attr._target_constraint[platform_common.ConstraintValueInfo]
    exec_constraint = ctx.attr._exec_constraint[platform_common.ConstraintValueInfo]
    host_constraint = ctx.attr._host_constraint[platform_common.ConstraintValueInfo]
    alpha_constraint = ctx.attr._alpha_constraint[platform_common.ConstraintValueInfo]
    beta_constraint = ctx.attr._beta_constraint[platform_common.ConstraintValueInfo]
    type = "unknown"
    if ctx.target_platform_has_constraint(target_constraint):
        type = "target"
    elif ctx.target_platform_has_constraint(exec_constraint):
        type = "exec"
    elif ctx.target_platform_has_constraint(host_constraint):
        type = "host"
    level = None
    if ctx.target_platform_has_constraint(alpha_constraint):
        level = "alpha"
    elif ctx.target_platform_has_constraint(beta_constraint):
        level = "beta"

    return [ShowPlatformInfo(
        platform_type = type,
        level = level,
    )]

show_platform = rule(
    implementation = _show_platform_impl,
    attrs = {
        "_target_constraint": attr.label(default = Label("//constraint:target")),
        "_exec_constraint": attr.label(default = Label("//constraint:exec")),
        "_host_constraint": attr.label(default = Label("//constraint:host")),
        "_alpha_constraint": attr.label(default = Label("//constraint:alpha")),
        "_beta_constraint": attr.label(default = Label("//constraint:beta")),
    },
    provides = [ShowPlatformInfo],
)
EOF
  cat >platform/BUILD.bazel <<EOF
load(":platform.bzl", "show_platform")

package(default_visibility = ["//visibility:public"])

show_platform(name = "platform")

# Common platforms.

# Execution platforms.
platform(
    name = "exec_alpha",
    constraint_values = [
        "//constraint:alpha",
        "//constraint:exec",
    ],
)

platform(
    name = "exec_beta",
    constraint_values = [
        "//constraint:beta",
        "//constraint:exec",
    ],
)

# Host platform.
platform(
    name = "host",
    constraint_values = [
        "//constraint:host",
    ],
)

# Target platform.
platform(
    name = "target",
    constraint_values = [
        "//constraint:target",
    ],
)
EOF
  # Append to WORKSPACE
  cat >>WORKSPACE <<EOF
register_execution_platforms(
    "//platform:exec_alpha",
    "//platform:exec_beta",
)
EOF
}

function write_toolchains() {
  mkdir toolchain
  cat >toolchain/toolchain.bzl <<EOF
load("//platform:platform.bzl", "ShowPlatformInfo", "describe_platform_info")
load(":extra_lib.bzl", "ExtraMessageProvider")

def _sample_toolchain_impl(ctx):
    target_dep = describe_platform_info(ctx.attr._target_dep[ShowPlatformInfo])
    tool_dep = describe_platform_info(ctx.attr._tool_dep[ShowPlatformInfo])
    dep_messages = []
    for dep in ctx.attr.deps:
        message = dep[ExtraMessageProvider].message
        dep_messages.append(message)
    message = "sample_toolchain: message: %s, target_dep: %s, tool_dep: %s, extra_deps: [%s]" % (
        ctx.attr.message,
        target_dep,
        tool_dep,
        ", ".join(dep_messages),
    )

    toolchain = platform_common.ToolchainInfo(
        message = message,
    )
    return [toolchain]

sample_toolchain = rule(
    implementation = _sample_toolchain_impl,
    attrs = {
        "message": attr.string(),
        "deps": attr.label_list(),
        "_target_dep": attr.label(cfg = "target", providers = [ShowPlatformInfo], default = Label("//platform")),
        "_tool_dep": attr.label(cfg = "exec", providers = [ShowPlatformInfo], default = Label("//platform")),
    },
)
EOF
  cat >toolchain/extra_lib.bzl <<EOF
load("//platform:platform.bzl", "ShowPlatformInfo", "describe_platform_info")

ExtraMessageProvider = provider(fields = ["message"])

def _extra_lib_impl(ctx):
    target_dep = describe_platform_info(ctx.attr._target_dep[ShowPlatformInfo])
    tool_dep = describe_platform_info(ctx.attr._tool_dep[ShowPlatformInfo])
    message = "extra_lib: message: %s, target_dep: %s, tool_dep: %s" % (
        ctx.attr.message,
        target_dep,
        tool_dep,
    )
    provider = ExtraMessageProvider(message = message)

    return [provider]

extra_lib = rule(
    implementation = _extra_lib_impl,
    attrs = {
        "message": attr.string(),
        "_target_dep": attr.label(cfg = "target", providers = [ShowPlatformInfo], default = Label("//platform")),
        "_tool_dep": attr.label(cfg = "exec", providers = [ShowPlatformInfo], default = Label("//platform")),
    },
    provides = [ExtraMessageProvider],
)
EOF
  cat >toolchain/BUILD.bazel <<EOF
package(default_visibility = ["//visibility:public"])

load(":toolchain.bzl", "sample_toolchain")

toolchain_type(name = "toolchain_type")

sample_toolchain(
    name = "sample_toolchain_alpha_impl",
    message = "alpha toolchain",
    deps = [
        ":extra",
    ],
)

toolchain(
    name = "sample_toolchain_alpha",
    exec_compatible_with = [
        "//constraint:alpha",
    ],
    toolchain = "sample_toolchain_alpha_impl",
    toolchain_type = ":toolchain_type",
)

sample_toolchain(
    name = "sample_toolchain_beta_impl",
    message = "beta toolchain",
    deps = [
        ":extra",
    ],
)

load(":extra_lib.bzl", "extra_lib")

extra_lib(
    name = "extra",
    message = "extra_lib foo",
)

toolchain(
    name = "sample_toolchain_beta",
    exec_compatible_with = [
        "//constraint:beta",
    ],
    toolchain = "sample_toolchain_beta_impl",
    toolchain_type = ":toolchain_type",
)
EOF
  # Append to WORKSPACE
  cat >>WORKSPACE <<EOF
register_toolchains(
    "//toolchain:sample_toolchain_alpha",
    "//toolchain:sample_toolchain_beta",
)
EOF
}

function write_rule() {
  mkdir rule
  cat >rule/rule.bzl <<EOF
load("//platform:platform.bzl", "ShowPlatformInfo", "describe_platform_info")

def _sample_impl(ctx):
    toolchain = ctx.toolchains["//toolchain:toolchain_type"]
    message = ctx.attr.message
    exec_platform = describe_platform_info(ctx.attr._exec[ShowPlatformInfo])

    str = 'Using toolchain: rule message: "%s", exec platform: "%s", toolchain message: "%s"\n' % (message, exec_platform, toolchain.message)

    log = ctx.outputs.log
    ctx.actions.write(
        output = log,
        content = str,
    )
    return [DefaultInfo(files = depset([log]))]

sample = rule(
    implementation = _sample_impl,
    attrs = {
        "message": attr.string(),
        "_exec": attr.label(cfg = "exec", providers = [ShowPlatformInfo], default = Label("//platform")),
    },
    outputs = {
        "log": "%{name}.log",
    },
    toolchains = ["//toolchain:toolchain_type"],
    incompatible_use_toolchain_transition = True,
)
EOF
  cat >rule/BUILD.bazel <<EOF
package(default_visibility = ["//visibility:public"])

EOF
}

function test_toolchain_transition() {
  write_constraints
  write_platforms
  write_toolchains
  write_rule

  cat >BUILD.bazel <<EOF
package(default_visibility = ["//visibility:public"])

load("//rule:rule.bzl", "sample")

sample(
    name = "sample",
    exec_compatible_with = [
        "//constraint:beta",
    ],
    message = "Hello",
)
EOF

  bazel build \
    --platforms=//platform:target \
    --host_platform=//platform:host \
     //:sample &> $TEST_log || fail "Build failed"

  # Verify contents of sample.log.
  cat bazel-bin/sample.log >> $TEST_log
  # The execution platform should be beta.
  expect_log 'rule message: "Hello", exec platform: "exec-beta"'
  # The toolchain should have proper target and exec matching the top target.
  expect_log 'sample_toolchain: message: beta toolchain, target_dep: target, tool_dep: exec-beta'
  # The toolchain's dependencies should use alpha for exec.
  # Make sure the exec platform does not propagate to further dependencies.
  expect_log 'extra_lib: message: extra_lib foo, target_dep: target, tool_dep: exec-alpha'
}

# Regression test for https://github.com/bazelbuild/bazel/issues/11993
# This was causing cquery to not correctly generate ConfiguredTargetKeys for
# toolchains, leading to the message "Targets were missing from graph"
function test_toolchain_transition_cquery() {
  write_constraints
  write_platforms
  write_toolchains
  write_rule

  cat >BUILD.bazel <<EOF
package(default_visibility = ["//visibility:public"])

load("//rule:rule.bzl", "sample")

sample(
    name = "sample",
    message = "Hello",
)
EOF

  bazel cquery \
    --platforms=//platform:target \
    --host_platform=//platform:host \
     'deps(//:sample)' &> $TEST_log || fail "Build failed"

  expect_not_log "Targets were missing from graph"
}

run_suite "toolchain transition tests"
