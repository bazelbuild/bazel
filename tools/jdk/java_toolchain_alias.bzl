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

"""Experimental re-implementations of Java toolchain aliases using toolchain resolution."""

def _java_runtime_alias(ctx):
    """An experimental implementation of java_runtime_alias using toolchain resolution."""
    toolchain_info = None
    if java_common.is_java_toolchain_resolution_enabled_do_not_use(ctx = ctx):
        toolchain_info = ctx.toolchains["@bazel_tools//tools/jdk:runtime_toolchain_type"]
        if hasattr(toolchain_info, "java_runtime"):
            toolchain = toolchain_info.java_runtime
        else:
            toolchain = toolchain_info
    else:
        toolchain = ctx.attr._java_runtime[java_common.JavaRuntimeInfo]
    providers = [
        toolchain,
        platform_common.TemplateVariableInfo({
            "JAVA": str(toolchain.java_executable_exec_path),
            "JAVABASE": str(toolchain.java_home),
        }),
        # See b/65239471 for related discussion of handling toolchain runfiles/data.
        DefaultInfo(
            runfiles = ctx.runfiles(transitive_files = toolchain.files),
            files = toolchain.files,
        ),
    ]
    if toolchain_info != None and toolchain_info != toolchain:
        providers.append(toolchain_info)
    return providers

java_runtime_alias = rule(
    implementation = _java_runtime_alias,
    toolchains = ["@bazel_tools//tools/jdk:runtime_toolchain_type"],
    incompatible_use_toolchain_transition = True,
    attrs = {
        "_java_runtime": attr.label(
            default = Label("@bazel_tools//tools/jdk:legacy_current_java_runtime"),
        ),
    },
)

def _java_host_runtime_alias(ctx):
    """An experimental implementation of java_host_runtime_alias using toolchain resolution."""
    runtime = ctx.attr._runtime
    java_runtime = runtime[java_common.JavaRuntimeInfo]
    template_variable_info = runtime[platform_common.TemplateVariableInfo]
    toolchain_info = platform_common.ToolchainInfo(java_runtime = java_runtime)
    return [
        java_runtime,
        template_variable_info,
        toolchain_info,
        # Create a new DefaultInfo instead of propagating runtime[DefaultInfo]
        # directly.
        DefaultInfo(
            files = runtime[DefaultInfo].files,
            data_runfiles = runtime[DefaultInfo].data_runfiles,
            default_runfiles = runtime[DefaultInfo].default_runfiles,
        ),
    ]

java_host_runtime_alias = rule(
    implementation = _java_host_runtime_alias,
    attrs = {
        "_runtime": attr.label(
            default = Label("@bazel_tools//tools/jdk:current_java_runtime"),
            providers = [
                java_common.JavaRuntimeInfo,
                platform_common.TemplateVariableInfo,
            ],
            cfg = "host",
        ),
    },
    provides = [
        java_common.JavaRuntimeInfo,
        platform_common.TemplateVariableInfo,
        platform_common.ToolchainInfo,
    ],
)

def _java_runtime_version_alias(ctx):
    """An alias fixing a specific version of java_runtime."""
    toolchain_info = None
    if java_common.is_java_toolchain_resolution_enabled_do_not_use(ctx = ctx):
        toolchain_info = ctx.toolchains["@bazel_tools//tools/jdk:runtime_toolchain_type"]
        if hasattr(toolchain_info, "java_runtime"):
            toolchain = toolchain_info.java_runtime
        else:
            toolchain = toolchain_info
    else:
        toolchain = ctx.attr.selected_java_runtime[java_common.JavaRuntimeInfo]
    providers = [
        toolchain,
        platform_common.TemplateVariableInfo({
            "JAVA": str(toolchain.java_executable_exec_path),
            "JAVABASE": str(toolchain.java_home),
        }),
        # See b/65239471 for related discussion of handling toolchain runfiles/data.
        DefaultInfo(
            runfiles = ctx.runfiles(transitive_files = toolchain.files),
            files = toolchain.files,
        ),
    ]
    if toolchain_info != None and toolchain_info != toolchain:
        providers.append(toolchain_info)
    return providers

def _java_runtime_transition_impl(settings, attr):
    return {"//command_line_option:java_runtime_version": attr.runtime_version}

_java_runtime_transition = transition(
    implementation = _java_runtime_transition_impl,
    inputs = [],
    outputs = ["//command_line_option:java_runtime_version"],
)

java_runtime_version_alias = rule(
    implementation = _java_runtime_version_alias,
    toolchains = ["@bazel_tools//tools/jdk:runtime_toolchain_type"],
    incompatible_use_toolchain_transition = True,
    attrs = {
        "runtime_version": attr.string(mandatory = True),
        # TODO(ilist): remove after java toolchain resolution flag is flipped
        "selected_java_runtime": attr.label(mandatory = True),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
    cfg = _java_runtime_transition,
)

def _java_toolchain_alias(ctx):
    """An experimental implementation of java_toolchain_alias using toolchain resolution."""
    toolchain_info = None
    if java_common.is_java_toolchain_resolution_enabled_do_not_use(ctx = ctx):
        toolchain_info = ctx.toolchains["@bazel_tools//tools/jdk:toolchain_type"]
        if hasattr(toolchain_info, "java"):
            toolchain = toolchain_info.java
        else:
            toolchain = toolchain_info
    else:
        toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo]
    providers = [toolchain]
    if toolchain_info != None and toolchain_info != toolchain:
        providers.append(toolchain_info)
    return struct(
        providers = providers,
        # Use the legacy provider syntax for compatibility with the native rules.
        java_toolchain = toolchain,
    )

java_toolchain_alias = rule(
    implementation = _java_toolchain_alias,
    toolchains = ["@bazel_tools//tools/jdk:toolchain_type"],
    attrs = {
        "_java_toolchain": attr.label(
            default = Label("@bazel_tools//tools/jdk:legacy_current_java_toolchain"),
        ),
    },
    incompatible_use_toolchain_transition = True,
)

# Add aliases for the legacy native rules to allow referring to both versions in @bazel_tools//tools/jdk
legacy_java_toolchain_alias = native.java_toolchain_alias
legacy_java_runtime_alias = native.java_runtime_alias
