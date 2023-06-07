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

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  if [[ -f "$0.runfiles_manifest" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
  elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
    export RUNFILES_DIR="$0.runfiles"
  fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  create_new_workspace
}

function write_constraints() {
  local pkg="${1}"
  mkdir -p "${pkg}/constraint"
  cat > "${pkg}/constraint/BUILD" <<EOF
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
  local pkg="${1}"
  mkdir -p "${pkg}/platform"

  cat > "${pkg}/platform/platform.bzl" <<EOF
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
        "_target_constraint": attr.label(default = Label("//${pkg}/constraint:target")),
        "_exec_constraint": attr.label(default = Label("//${pkg}/constraint:exec")),
        "_host_constraint": attr.label(default = Label("//${pkg}/constraint:host")),
        "_alpha_constraint": attr.label(default = Label("//${pkg}/constraint:alpha")),
        "_beta_constraint": attr.label(default = Label("//${pkg}/constraint:beta")),
    },
    provides = [ShowPlatformInfo],
)
EOF
  cat > "${pkg}/platform/BUILD" <<EOF
load(":platform.bzl", "show_platform")

package(default_visibility = ["//visibility:public"])

show_platform(name = "platform")

# Common platforms.

# Execution platforms.
platform(
    name = "exec_alpha",
    constraint_values = [
        "//${pkg}/constraint:alpha",
        "//${pkg}/constraint:exec",
    ],
)

platform(
    name = "exec_beta",
    constraint_values = [
        "//${pkg}/constraint:beta",
        "//${pkg}/constraint:exec",
    ],
)

# Host platform.
platform(
    name = "host",
    constraint_values = [
        "//${pkg}/constraint:host",
    ],
)

# Target platform.
platform(
    name = "target",
    constraint_values = [
        "//${pkg}/constraint:target",
    ],
)
EOF

  # Append to WORKSPACE
  cat >>WORKSPACE <<EOF
register_execution_platforms(
    "//${pkg}/platform:exec_alpha",
    "//${pkg}/platform:exec_beta",
)
EOF
}

function write_toolchains() {
  local pkg="${1}"
  mkdir -p "${pkg}/toolchain"
  cat > "${pkg}/toolchain/toolchain.bzl" <<EOF
load("//${pkg}/platform:platform.bzl", "ShowPlatformInfo", "describe_platform_info")
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
        "_target_dep": attr.label(
            cfg = "target",
            providers = [ShowPlatformInfo],
            default = Label("//${pkg}/platform")),
        "_tool_dep": attr.label(
            cfg = "exec",
            providers = [ShowPlatformInfo],
            default = Label("//${pkg}/platform")),
    },
)
EOF
  cat > "${pkg}/toolchain/extra_lib.bzl" <<EOF
load("//${pkg}/platform:platform.bzl", "ShowPlatformInfo", "describe_platform_info")

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
        "_target_dep": attr.label(
            cfg = "target",
            providers = [ShowPlatformInfo],
            default = Label("//${pkg}/platform")),
        "_tool_dep": attr.label(
            cfg = "exec",
            providers = [ShowPlatformInfo],
            default = Label("//${pkg}/platform")),
    },
    provides = [ExtraMessageProvider],
)
EOF
  cat > "${pkg}/toolchain/BUILD" <<EOF
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
        "//${pkg}/constraint:alpha",
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
        "//${pkg}/constraint:beta",
    ],
    toolchain = "sample_toolchain_beta_impl",
    toolchain_type = ":toolchain_type",
)
EOF

  # Append to WORKSPACE
  cat >>WORKSPACE <<EOF
register_toolchains(
    "//${pkg}/toolchain:sample_toolchain_alpha",
    "//${pkg}/toolchain:sample_toolchain_beta",
)
EOF
}

function write_rules() {
  local pkg="${1}"
  mkdir -p "${pkg}/rule"
  cat > "${pkg}/rule/rule.bzl" <<EOF
load("//${pkg}/platform:platform.bzl", "ShowPlatformInfo", "describe_platform_info")

def _report(ctx):
    toolchain = ctx.toolchains["//${pkg}/toolchain:toolchain_type"]
    message = ctx.attr.message
    exec_platform = describe_platform_info(ctx.attr._exec[ShowPlatformInfo])

    str = 'Using toolchain: rule message: "%s", exec platform: "%s", toolchain message: "%s"\n' % (message, exec_platform, toolchain.message)

    log = ctx.outputs.log
    ctx.actions.write(
        output = log,
        content = str,
    )
    return log

def _sample_impl(ctx):
    log = _report(ctx)
    return [DefaultInfo(files = depset([log]))]

sample = rule(
    implementation = _sample_impl,
    attrs = {
        "message": attr.string(),
        "_exec": attr.label(
            cfg = "exec",
            providers = [ShowPlatformInfo],
            default = Label("//${pkg}/platform")),
    },
    outputs = {
        "log": "%{name}.log",
    },
    toolchains = ["//${pkg}/toolchain:toolchain_type"],
    incompatible_use_toolchain_transition = True,
)

def _sample_test_impl(ctx):
    log = _report(ctx)
    # Write a fake test executable.
    executable = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(
        output = executable,
        content = "\n".join([
            "#!/usr/bin/env bash",
            "exit 0",
        ]),
        is_executable = True,
    )

    return [DefaultInfo(
        executable = executable,
        files = depset([executable, log]),
    )]

sample_test = rule(
    implementation = _sample_test_impl,
    attrs = {
        "message": attr.string(),
        "_exec": attr.label(
            cfg = "exec",
            providers = [ShowPlatformInfo],
            default = Label("//${pkg}/platform")),
    },
    outputs = {
        "log": "%{name}.log",
    },
    toolchains = ["//${pkg}/toolchain:toolchain_type"],
    incompatible_use_toolchain_transition = True,
    test = True,
)
EOF
  cat > "${pkg}/rule/BUILD" <<EOF
package(default_visibility = ["//visibility:public"])

EOF
}

function test_toolchain_transition() {
  local -r pkg="${FUNCNAME[0]}"
  write_constraints "${pkg}"
  write_platforms "${pkg}"
  write_toolchains "${pkg}"
  write_rules "${pkg}"

  mkdir -p "${pkg}"
  cat > "${pkg}/BUILD" <<EOF
package(default_visibility = ["//visibility:public"])

load("//${pkg}/rule:rule.bzl", "sample")

sample(
    name = "sample",
    exec_compatible_with = [
        "//${pkg}/constraint:beta",
    ],
    message = "Hello",
)
EOF

  bazel build \
    --platforms="//${pkg}/platform:target" \
    --host_platform="//${pkg}/platform:host" \
     "//${pkg}:sample" &> $TEST_log || fail "Build failed"

  # Verify contents of sample.log.
  cat "bazel-bin/${pkg}/sample.log" >> $TEST_log
  # The execution platform should be beta.
  expect_log 'rule message: "Hello", exec platform: "exec-beta"'
  # The toolchain should have proper target and exec matching the top target.
  expect_log 'sample_toolchain: message: beta toolchain, target_dep: target, tool_dep: exec-beta'
  # The toolchain's dependencies should use alpha for exec.
  # Make sure the exec platform does not propagate to further dependencies.
  expect_log 'extra_lib: message: extra_lib foo, target_dep: target, tool_dep: exec-alpha'
}

# Regression test for b/284495846.
# Test rules use the test trimming transition, which means that toolchain
# dependencies are in a different configuration that the actual parent target,
# which caused an issue where the execution platform was not correctly forwarded
# to the toolchain.
function test_toolchain_transition_test_rule() {
  local -r pkg="${FUNCNAME[0]}"
  write_constraints "${pkg}"
  write_platforms "${pkg}"
  write_toolchains "${pkg}"
  write_rules "${pkg}"

  mkdir -p "${pkg}"
  cat > "${pkg}/BUILD" <<EOF
package(default_visibility = ["//visibility:public"])

load("//${pkg}/rule:rule.bzl", "sample_test")

# Use a test rul to trigger b/284495846
sample_test(
    name = "sample_test",
    exec_compatible_with = [
        "//${pkg}/constraint:beta",
    ],
    message = "Hello",
)
EOF

  bazel build \
    --platforms="//${pkg}/platform:target" \
    --host_platform="//${pkg}/platform:host" \
    --use_target_platform_for_tests \
     "//${pkg}:sample_test" &> $TEST_log || fail "Build failed"

  # Verify contents of sample_test.log.
  cat "bazel-bin/${pkg}/sample_test.log" >> $TEST_log
  # The execution platform should be beta.
  expect_log 'rule message: "Hello", exec platform: "exec-beta"'
  # The toolchain should have proper target and exec matching the top target.
  expect_log 'sample_toolchain: message: beta toolchain, target_dep: target, tool_dep: exec-beta'
}

# Regression test for https://github.com/bazelbuild/bazel/issues/11993
# This was causing cquery to not correctly generate ConfiguredTargetKeys for
# toolchains, leading to the message "Targets were missing from graph"
function test_toolchain_transition_cquery() {
  local -r pkg="${FUNCNAME[0]}"
  write_constraints "${pkg}"
  write_platforms "${pkg}"
  write_toolchains "${pkg}"
  write_rules "${pkg}"

  mkdir -p "${pkg}"
  cat > "${pkg}/BUILD" <<EOF
package(default_visibility = ["//visibility:public"])

load("//${pkg}/rule:rule.bzl", "sample")

sample(
    name = "sample",
    message = "Hello",
)
EOF

  bazel cquery \
    --platforms="//${pkg}/platform:target" \
    --host_platform="//${pkg}/platform:host" \
     "deps(//${pkg}:sample)" &> $TEST_log || fail "Build failed"

  expect_not_log "Targets were missing from graph"
}

run_suite "toolchain transition tests"
