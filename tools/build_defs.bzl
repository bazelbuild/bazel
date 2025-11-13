# Copyright 2024 The Bazel Authors. All rights reserved.
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

"""Utilities for managing tools for different platforms."""

load("@platforms//host:constraints.bzl", "HOST_CONSTRAINTS")

visibility("//tools/...")

BZLMOD_ENABLED = str(Label("@bazel_tools//:foo")).startswith("@@")

IS_HOST_WINDOWS = Label("@platforms//os:windows") in [Label(label) for label in HOST_CONSTRAINTS]

def _single_binary_toolchain_rule_impl(ctx):
    return platform_common.ToolchainInfo(
        binary = ctx.file.binary,
    )

_single_binary_toolchain_rule = rule(
    implementation = _single_binary_toolchain_rule_impl,
    attrs = {
        "binary": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
    },
)

def single_binary_toolchain(
        *,
        name,
        toolchain_type,
        binary = None,
        target_compatible_with = [],
        exec_compatible_with = []):
    """Declares a toolchain together with its implementation for an optional single binary."""
    impl_name = name + "_impl"

    _single_binary_toolchain_rule(
        name = impl_name,
        binary = binary,
        # Avoid eager loading of the binary, which may come from a remote
        # repository, in wildcard builds.
        tags = ["manual"],
        visibility = ["//visibility:private"],
    )

    native.toolchain(
        name = name,
        toolchain_type = toolchain_type,
        toolchain = ":" + impl_name,
        target_compatible_with = target_compatible_with,
        exec_compatible_with = exec_compatible_with,
        visibility = ["//visibility:private"],
    )

def _current_toolchain_base_impl(ctx, *, toolchain_type):
    executable = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.symlink(
        output = executable,
        target_file = ctx.toolchains[toolchain_type].binary,
    )
    return DefaultInfo(executable = executable)

def _make_current_toolchain_rule(toolchain_type):
    return rule(
        implementation = lambda ctx: _current_toolchain_base_impl(ctx, toolchain_type = toolchain_type),
        toolchains = [toolchain_type],
        executable = True,
    )

current_launcher_binary = _make_current_toolchain_rule("//tools/launcher:launcher_toolchain_type")
