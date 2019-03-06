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

""" A rule that mocks cc_toolchain configuration."""

load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    _action_config = "action_config",
    _artifact_name_pattern = "artifact_name_pattern",
    _env_entry = "env_entry",
    _env_set = "env_set",
    _feature = "feature",
    _feature_set = "feature_set",
    _flag_group = "flag_group",
    _flag_set = "flag_set",
    _make_variable = "make_variable",
    _tool = "tool",
    _tool_path = "tool_path",
    _variable_with_value = "variable_with_value",
    _with_feature_set = "with_feature_set",
)
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

action_config = _action_config
artifact_name_pattern = _artifact_name_pattern
env_entry = _env_entry
env_set = _env_set
feature = _feature
feature_set = _feature_set
flag_group = _flag_group
flag_set = _flag_set
make_variable = _make_variable
tool = _tool
tool_path = _tool_path
variable_with_value = _variable_with_value
with_feature_set = _with_feature_set

def _impl(ctx):
    toolchain_identifier = "mock-llvm-toolchain-k8"

    host_system_name = "local"

    target_system_name = "local"

    target_cpu = "k8"

    target_libc = "local"

    compiler = "compiler"

    abi_version = "local"

    abi_libc_version = "local"

    cc_target_os = None

    builtin_sysroot = "/usr/grte/v1"

    all_link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

    action_configs = []

    default_compile_flags_feature = feature(
        name = "default_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = [flag_group(flags = ["--default-compile-flag"])],
            ),
        ],
    )

    default_link_flags_feature = feature(
        name = "default_link_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["-ldefault-link-flag"])],
            ),
        ],
    )

    features = [default_compile_flags_feature, default_link_flags_feature]

    cxx_builtin_include_directories = ["/usr/lib/gcc/", "/usr/local/include", "/usr/include"]

    artifact_name_patterns = []

    make_variables = []

    tool_paths = [
        tool_path(name = "ar", path = "/usr/bin/mock-ar"),
        tool_path(
            name = "compat-ld",
            path = "/usr/bin/mock-compat-ld",
        ),
        tool_path(name = "cpp", path = "/usr/bin/mock-cpp"),
        tool_path(name = "dwp", path = "/usr/bin/mock-dwp"),
        tool_path(name = "gcc", path = "/usr/bin/mock-gcc"),
        tool_path(name = "gcov", path = "/usr/bin/mock-gcov"),
        tool_path(name = "ld", path = "/usr/bin/mock-ld"),
        tool_path(name = "nm", path = "/usr/bin/mock-nm"),
        tool_path(name = "objcopy", path = "/usr/bin/mock-objcopy"),
        tool_path(name = "objdump", path = "/usr/bin/mock-objdump"),
        tool_path(name = "strip", path = "/usr/bin/mock-strip"),
        tool_path(
            name = "llvm-profdata",
            path = "/usr/bin/mock-llvm-profdata",
        ),
    ]

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
        "cpu": attr.string(),
        "compiler": attr.string(),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
