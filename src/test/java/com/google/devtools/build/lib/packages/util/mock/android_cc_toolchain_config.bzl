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
"""Mock cc_toolchain_config rule for Android ndk tests."""

load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "artifact_name_pattern",
    "feature",
    "flag_group",
    "flag_set",
    "make_variable",
    "tool_path",
)

_FEATURE_NAMES = struct(
    linktest = "linktest",
    static_link_cpp_runtimes = "static_link_cpp_runtimes",
    supports_pic = "supports_pic",
)

_linktest_feature = feature(
    name = _FEATURE_NAMES.linktest,
    flag_sets = [
        flag_set(
            actions = ["c++-link-dynamic-library"],
            flag_groups = [
                flag_group(
                    flags = ["--winnie_the_pooh"],
                ),
            ],
        ),
    ],
)

_supports_pic_feature = feature(name = _FEATURE_NAMES.supports_pic, enabled = True)

_static_link_cpp_runtimes_feature = feature(
    name = _FEATURE_NAMES.static_link_cpp_runtimes,
    enabled = True,
)

_feature_name_to_feature = {
    _FEATURE_NAMES.linktest: _linktest_feature,
    _FEATURE_NAMES.static_link_cpp_runtimes: _static_link_cpp_runtimes_feature,
    _FEATURE_NAMES.supports_pic: _supports_pic_feature,
}

_action_name_to_action = {}

def _get_artifact_name_pattern(category, prefix, extension):
    return artifact_name_pattern(
        category_name = category,
        prefix = prefix,
        extension = extension,
    )

def _impl(ctx):
    toolchain_identifier = ctx.attr.toolchain_identifier
    host_system_name = ctx.attr.host_system_name
    target_system_name = ctx.attr.target_system_name
    target_cpu = ctx.attr.cpu
    target_libc = ctx.attr.target_libc
    compiler = ctx.attr.compiler
    abi_version = ctx.attr.abi_version
    abi_libc_version = ctx.attr.abi_libc_version
    cc_target_os = ctx.attr.cc_target_os if ctx.attr.cc_target_os != "" else None
    builtin_sysroot = ctx.attr.builtin_sysroot if ctx.attr.builtin_sysroot != "" else None

    artifact_name_patterns = []

    for category, values in ctx.attr.artifact_name_patterns.items():
        artifact_name_patterns.append(_get_artifact_name_pattern(category, values[0], values[1]))

    action_configs = [_action_name_to_action[name] for name in ctx.attr.action_configs]

    make_variables = [
        make_variable(name = name, value = value)
        for name, value in ctx.attr.make_variables.items()
    ]

    tool_paths = [tool_path(name, path) for name, path in ctx.attr.tool_paths.items()]

    features = [_feature_name_to_feature[name] for name in ctx.attr.feature_names]

    cxx_builtin_include_directories = ctx.attr.cxx_builtin_include_directories

    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(out, "Fake executable")
    return [
        cc_common.create_cc_toolchain_config_info(
            ctx = ctx,
            features = features,
            action_configs = action_configs,
            artifact_name_patterns = artifact_name_patterns,
            cxx_builtin_include_directories = cxx_builtin_include_directories,
            toolchain_identifier = toolchain_identifier,
            host_system_name = host_system_name,
            target_system_name = target_system_name,
            target_cpu = target_cpu,
            target_libc = target_libc,
            compiler = compiler,
            abi_version = abi_version,
            abi_libc_version = abi_libc_version,
            tool_paths = tool_paths,
            make_variables = make_variables,
            builtin_sysroot = builtin_sysroot,
            cc_target_os = cc_target_os,
        ),
        DefaultInfo(
            executable = out,
        ),
    ]

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory = True, values = ["armeabi-v7a", "x86"]),
        "compiler": attr.string(default = "gcc"),
        "toolchain_identifier": attr.string(default = "x86"),
        "host_system_name": attr.string(default = "x86"),
        "target_system_name": attr.string(default = "x86-linux-android"),
        "target_libc": attr.string(default = "local"),
        "abi_version": attr.string(default = "x86"),
        "abi_libc_version": attr.string(default = "r7"),
        "feature_names": attr.string_list(),
        "action_configs": attr.string_list(),
        "artifact_name_patterns": attr.string_list_dict(),
        "cc_target_os": attr.string(),
        "builtin_sysroot": attr.string(),
        "tool_paths": attr.string_dict(),
        "cxx_builtin_include_directories": attr.string_list(),
        "make_variables": attr.string_dict(),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
