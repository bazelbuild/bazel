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
""" Mock objc toolchain configuration. """

load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "artifact_name_pattern",
    "env_entry",
    "env_set",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "make_variable",
    "tool",
    "tool_path",
    "variable_with_value",
    "with_feature_set",
)
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

# Link actions using the C++ bazel actions.
_NON_OBJC_LINK_ACTIONS = [
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

# Link actions using the ObjC bazel actions.
_OBJC_LINK_ACTIONS = [
    ACTION_NAMES.objc_executable,
    ACTION_NAMES.objcpp_executable,
]

_ALL_LINK_ACTIONS = _NON_OBJC_LINK_ACTIONS + _OBJC_LINK_ACTIONS

_default_feature = feature(
    name = "default",
    enabled = True,
    flag_sets = [
        flag_set(
            actions = ["objc-compile"],
            flag_groups = [flag_group(flags = ["-dummy"])],
        ),
    ],
)

_contains_objc_sources_feature = feature(
    name = "contains_objc_sources",
    flag_sets = [
        flag_set(
            actions = [ACTION_NAMES.cpp_compile],
            flag_groups = [flag_group(flags = ["DUMMY_FLAG"])],
        ),
    ],
)

_supports_interface_shared_libraries_feature = feature(
    name = "supports_interface_shared_libraries",
    enabled = True,
)

_supports_dynamic_linker_feature = feature(
    name = "supports_dynamic_linker",
    enabled = True,
)

_feature_name_to_feature = {
    "default_feature": _default_feature,
    "contains_objc_sources": _contains_objc_sources_feature,
    "supports_interface_shared_libraries": _supports_interface_shared_libraries_feature,
    "supports_dynamic_linker": _supports_dynamic_linker_feature,
}

_action_name_to_action = {}

def _get_artifact_name_pattern(category_name, prefix, extension):
    return artifact_name_pattern(
        category_name = category_name,
        prefix = prefix,
        extension = extension,
    )

def _impl(ctx):
    toolchain_identifier = ctx.attr.toolchain_identifier
    target_cpu = ctx.attr.cpu
    compiler = ctx.attr.compiler
    host_system_name = ctx.attr.host_system_name
    target_system_name = ctx.attr.target_system_name
    target_libc = ctx.attr.target_libc
    abi_version = ctx.attr.abi_version
    abi_libc_version = ctx.attr.abi_libc_version
    builtin_sysroot = ctx.attr.builtin_sysroot if ctx.attr.builtin_sysroot != "" else None
    cc_target_os = ctx.attr.cc_target_os if ctx.attr.cc_target_os != "" else None

    if (ctx.attr.cpu == "tvos_arm64"):
        preprocess_assemble_action = action_config(
            action_name = ACTION_NAMES.preprocess_assemble,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "unfiltered_cxx_flags",
            ],
            tools = [
                tool(
                    path = "tvos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        preprocess_assemble_action = action_config(
            action_name = ACTION_NAMES.preprocess_assemble,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "unfiltered_cxx_flags",
            ],
            tools = [
                tool(
                    path = "tvsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "x64_windows"):
        preprocess_assemble_action = action_config(
            action_name = ACTION_NAMES.preprocess_assemble,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "<tool_dir>/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7"):
        preprocess_assemble_action = action_config(
            action_name = ACTION_NAMES.preprocess_assemble,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64"):
        preprocess_assemble_action = action_config(
            action_name = ACTION_NAMES.preprocess_assemble,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        preprocess_assemble_action = action_config(
            action_name = ACTION_NAMES.preprocess_assemble,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "mac/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        preprocess_assemble_action = action_config(
            action_name = ACTION_NAMES.preprocess_assemble,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        preprocess_assemble_action = action_config(
            action_name = ACTION_NAMES.preprocess_assemble,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        preprocess_assemble_action = None

    if (ctx.attr.cpu == "x64_windows"):
        objc_executable_action = action_config(
            action_name = "objc-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-arch <architecture>"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "<tool_dir>/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64"):
        objc_executable_action = action_config(
            action_name = "objc-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-arch arm64"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        objc_executable_action = action_config(
            action_name = "objc-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-arch arm64"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "tvos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_armv7"):
        objc_executable_action = action_config(
            action_name = "objc-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-arch armv7"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        objc_executable_action = action_config(
            action_name = "objc-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-arch armv7k"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "watchos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386"):
        objc_executable_action = action_config(
            action_name = "objc-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-arch i386"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        objc_executable_action = action_config(
            action_name = "objc-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-arch i386"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "watchsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_x86_64"):
        objc_executable_action = action_config(
            action_name = "objc-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-arch x86_64"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        objc_executable_action = action_config(
            action_name = "objc-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-arch x86_64"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "mac/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        objc_executable_action = action_config(
            action_name = "objc-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-arch x86_64"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "tvsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        objc_executable_action = None

    if (ctx.attr.cpu == "x64_windows"):
        objcpp_executable_action = action_config(
            action_name = "objc++-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-stdlib=libc++", "-std=gnu++11"]),
                        flag_group(flags = ["-arch <architecture>"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "<tool_dir>/wrapped_clang++",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64"):
        objcpp_executable_action = action_config(
            action_name = "objc++-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-stdlib=libc++", "-std=gnu++11"]),
                        flag_group(flags = ["-arch arm64"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang++",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        objcpp_executable_action = action_config(
            action_name = "objc++-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-stdlib=libc++", "-std=gnu++11"]),
                        flag_group(flags = ["-arch arm64"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "tvos/wrapped_clang++",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_armv7"):
        objcpp_executable_action = action_config(
            action_name = "objc++-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-stdlib=libc++", "-std=gnu++11"]),
                        flag_group(flags = ["-arch armv7"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang++",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        objcpp_executable_action = action_config(
            action_name = "objc++-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-stdlib=libc++", "-std=gnu++11"]),
                        flag_group(flags = ["-arch armv7k"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "watchos/wrapped_clang++",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386"):
        objcpp_executable_action = action_config(
            action_name = "objc++-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-stdlib=libc++", "-std=gnu++11"]),
                        flag_group(flags = ["-arch i386"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang++",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        objcpp_executable_action = action_config(
            action_name = "objc++-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-stdlib=libc++", "-std=gnu++11"]),
                        flag_group(flags = ["-arch i386"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "watchsim/wrapped_clang++",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_x86_64"):
        objcpp_executable_action = action_config(
            action_name = "objc++-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-stdlib=libc++", "-std=gnu++11"]),
                        flag_group(flags = ["-arch x86_64"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang++",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        objcpp_executable_action = action_config(
            action_name = "objc++-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-stdlib=libc++", "-std=gnu++11"]),
                        flag_group(flags = ["-arch x86_64"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "mac/wrapped_clang++",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        objcpp_executable_action = action_config(
            action_name = "objc++-executable",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-stdlib=libc++", "-std=gnu++11"]),
                        flag_group(flags = ["-arch x86_64"]),
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-objc_abi_version",
                                "-Xlinker",
                                "2",
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                "@executable_path/Frameworks",
                                "-fobjc-link-runtime",
                                "-ObjC",
                            ],
                        ),
                        flag_group(
                            flags = ["-framework %{framework_names}"],
                            iterate_over = "framework_names",
                        ),
                        flag_group(
                            flags = ["-weak_framework %{weak_framework_names}"],
                            iterate_over = "weak_framework_names",
                        ),
                        flag_group(
                            flags = ["-l%{library_names}"],
                            iterate_over = "library_names",
                        ),
                        flag_group(flags = ["-filelist %{filelist}"]),
                        flag_group(flags = ["-o %{linked_binary}"]),
                        flag_group(
                            flags = ["-force_load %{force_load_exec_paths}"],
                            iterate_over = "force_load_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{dep_linkopts}"],
                            iterate_over = "dep_linkopts",
                        ),
                        flag_group(
                            flags = ["-Wl,%{attr_linkopts}"],
                            iterate_over = "attr_linkopts",
                        ),
                    ],
                ),
            ],
            implies = [
                "include_system_dirs",
                "framework_paths",
                "version_min",
                "apple_env",
                "apply_implicit_frameworks",
            ],
            tools = [
                tool(
                    path = "tvsim/wrapped_clang++",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        objcpp_executable_action = None

    if (ctx.attr.cpu == "x64_windows"):
        cpp_link_executable_action = action_config(
            action_name = ACTION_NAMES.cpp_link_executable,
            implies = [
                "symbol_counts",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "force_pic_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "<tool_dir>/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7"):
        cpp_link_executable_action = action_config(
            action_name = ACTION_NAMES.cpp_link_executable,
            implies = [
                "symbol_counts",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "force_pic_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "ios/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64"):
        cpp_link_executable_action = action_config(
            action_name = ACTION_NAMES.cpp_link_executable,
            implies = [
                "symbol_counts",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "force_pic_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "iossim/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        cpp_link_executable_action = action_config(
            action_name = ACTION_NAMES.cpp_link_executable,
            implies = [
                "symbol_counts",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "force_pic_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "mac/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        cpp_link_executable_action = action_config(
            action_name = ACTION_NAMES.cpp_link_executable,
            implies = [
                "symbol_counts",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "force_pic_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "tvos/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        cpp_link_executable_action = action_config(
            action_name = ACTION_NAMES.cpp_link_executable,
            implies = [
                "symbol_counts",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "force_pic_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "tvsim/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        cpp_link_executable_action = action_config(
            action_name = ACTION_NAMES.cpp_link_executable,
            implies = [
                "symbol_counts",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "force_pic_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "watchos/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        cpp_link_executable_action = action_config(
            action_name = ACTION_NAMES.cpp_link_executable,
            implies = [
                "symbol_counts",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "force_pic_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "watchsim/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        cpp_link_executable_action = None

    if (ctx.attr.cpu == "x64_windows"):
        objc_compile_action = action_config(
            action_name = ACTION_NAMES.objc_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch <architecture>"])],
                ),
            ],
            implies = [
                "objc_actions",
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "<tool_dir>/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64"):
        objc_compile_action = action_config(
            action_name = ACTION_NAMES.objc_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch arm64"])],
                ),
            ],
            implies = [
                "objc_actions",
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        objc_compile_action = action_config(
            action_name = ACTION_NAMES.objc_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch arm64"])],
                ),
            ],
            implies = [
                "objc_actions",
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "tvos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_armv7"):
        objc_compile_action = action_config(
            action_name = ACTION_NAMES.objc_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch armv7"])],
                ),
            ],
            implies = [
                "objc_actions",
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        objc_compile_action = action_config(
            action_name = ACTION_NAMES.objc_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch armv7k"])],
                ),
            ],
            implies = [
                "objc_actions",
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386"):
        objc_compile_action = action_config(
            action_name = ACTION_NAMES.objc_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch i386"])],
                ),
            ],
            implies = [
                "objc_actions",
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "apply_simulator_compiler_flags",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        objc_compile_action = action_config(
            action_name = ACTION_NAMES.objc_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch i386"])],
                ),
            ],
            implies = [
                "objc_actions",
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "apply_simulator_compiler_flags",
            ],
            tools = [
                tool(
                    path = "watchsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_x86_64"):
        objc_compile_action = action_config(
            action_name = ACTION_NAMES.objc_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch x86_64"])],
                ),
            ],
            implies = [
                "objc_actions",
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "apply_simulator_compiler_flags",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        objc_compile_action = action_config(
            action_name = ACTION_NAMES.objc_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch x86_64"])],
                ),
            ],
            implies = [
                "objc_actions",
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "apply_simulator_compiler_flags",
            ],
            tools = [
                tool(
                    path = "tvsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        objc_compile_action = action_config(
            action_name = ACTION_NAMES.objc_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch x86_64"])],
                ),
            ],
            implies = [
                "objc_actions",
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "mac/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        objc_compile_action = None

    if (ctx.attr.cpu == "x64_windows"):
        objc_archive_action = action_config(
            action_name = "objc-archive",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-filelist",
                                "%{obj_list_path}",
                                "-arch_only",
                                "<architecture>",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{archive_path}",
                            ],
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "<tool_dir>/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64"):
        objc_archive_action = action_config(
            action_name = "objc-archive",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-filelist",
                                "%{obj_list_path}",
                                "-arch_only",
                                "arm64",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{archive_path}",
                            ],
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "ios/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        objc_archive_action = action_config(
            action_name = "objc-archive",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-filelist",
                                "%{obj_list_path}",
                                "-arch_only",
                                "arm64",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{archive_path}",
                            ],
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "tvos/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_armv7"):
        objc_archive_action = action_config(
            action_name = "objc-archive",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-filelist",
                                "%{obj_list_path}",
                                "-arch_only",
                                "armv7",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{archive_path}",
                            ],
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "ios/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        objc_archive_action = action_config(
            action_name = "objc-archive",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-filelist",
                                "%{obj_list_path}",
                                "-arch_only",
                                "armv7k",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{archive_path}",
                            ],
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "watchos/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386"):
        objc_archive_action = action_config(
            action_name = "objc-archive",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-filelist",
                                "%{obj_list_path}",
                                "-arch_only",
                                "i386",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{archive_path}",
                            ],
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "iossim/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        objc_archive_action = action_config(
            action_name = "objc-archive",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-filelist",
                                "%{obj_list_path}",
                                "-arch_only",
                                "i386",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{archive_path}",
                            ],
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "watchsim/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_x86_64"):
        objc_archive_action = action_config(
            action_name = "objc-archive",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-filelist",
                                "%{obj_list_path}",
                                "-arch_only",
                                "x86_64",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{archive_path}",
                            ],
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "iossim/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        objc_archive_action = action_config(
            action_name = "objc-archive",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-filelist",
                                "%{obj_list_path}",
                                "-arch_only",
                                "x86_64",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{archive_path}",
                            ],
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "mac/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        objc_archive_action = action_config(
            action_name = "objc-archive",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-filelist",
                                "%{obj_list_path}",
                                "-arch_only",
                                "x86_64",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{archive_path}",
                            ],
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "tvsim/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        objc_archive_action = None

    if (ctx.attr.cpu == "x64_windows"):
        objcpp_compile_action = action_config(
            action_name = ACTION_NAMES.objcpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = ["-arch <architecture>", "-stdlib=libc++", "-std=gnu++11"],
                        ),
                    ],
                ),
            ],
            implies = [
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "<tool_dir>/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64"):
        objcpp_compile_action = action_config(
            action_name = ACTION_NAMES.objcpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = ["-arch arm64", "-stdlib=libc++", "-std=gnu++11"],
                        ),
                    ],
                ),
            ],
            implies = [
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        objcpp_compile_action = action_config(
            action_name = ACTION_NAMES.objcpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = ["-arch arm64", "-stdlib=libc++", "-std=gnu++11"],
                        ),
                    ],
                ),
            ],
            implies = [
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "tvos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_armv7"):
        objcpp_compile_action = action_config(
            action_name = ACTION_NAMES.objcpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = ["-arch armv7", "-stdlib=libc++", "-std=gnu++11"],
                        ),
                    ],
                ),
            ],
            implies = [
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        objcpp_compile_action = action_config(
            action_name = ACTION_NAMES.objcpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = ["-arch armv7k", "-stdlib=libc++", "-std=gnu++11"],
                        ),
                    ],
                ),
            ],
            implies = [
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386"):
        objcpp_compile_action = action_config(
            action_name = ACTION_NAMES.objcpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = ["-arch i386", "-stdlib=libc++", "-std=gnu++11"],
                        ),
                    ],
                ),
            ],
            implies = [
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "apply_simulator_compiler_flags",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        objcpp_compile_action = action_config(
            action_name = ACTION_NAMES.objcpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = ["-arch i386", "-stdlib=libc++", "-std=gnu++11"],
                        ),
                    ],
                ),
            ],
            implies = [
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "apply_simulator_compiler_flags",
            ],
            tools = [
                tool(
                    path = "watchsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_x86_64"):
        objcpp_compile_action = action_config(
            action_name = ACTION_NAMES.objcpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = ["-arch x86_64", "-stdlib=libc++", "-std=gnu++11"],
                        ),
                    ],
                ),
            ],
            implies = [
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "apply_simulator_compiler_flags",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        objcpp_compile_action = action_config(
            action_name = ACTION_NAMES.objcpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = ["-arch x86_64", "-stdlib=libc++", "-std=gnu++11"],
                        ),
                    ],
                ),
            ],
            implies = [
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "apply_simulator_compiler_flags",
            ],
            tools = [
                tool(
                    path = "tvsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        objcpp_compile_action = action_config(
            action_name = ACTION_NAMES.objcpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = ["-arch x86_64", "-stdlib=libc++", "-std=gnu++11"],
                        ),
                    ],
                ),
            ],
            implies = [
                "apply_default_compiler_flags",
                "apply_default_warnings",
                "framework_paths",
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "mac/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        objcpp_compile_action = None

    if (ctx.attr.cpu == "tvos_arm64"):
        cpp_header_parsing_action = action_config(
            action_name = ACTION_NAMES.cpp_header_parsing,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "unfiltered_cxx_flags",
            ],
            tools = [
                tool(
                    path = "tvos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        cpp_header_parsing_action = action_config(
            action_name = ACTION_NAMES.cpp_header_parsing,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "unfiltered_cxx_flags",
            ],
            tools = [
                tool(
                    path = "tvsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "x64_windows"):
        cpp_header_parsing_action = action_config(
            action_name = ACTION_NAMES.cpp_header_parsing,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "<tool_dir>/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7"):
        cpp_header_parsing_action = action_config(
            action_name = ACTION_NAMES.cpp_header_parsing,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64"):
        cpp_header_parsing_action = action_config(
            action_name = ACTION_NAMES.cpp_header_parsing,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        cpp_header_parsing_action = action_config(
            action_name = ACTION_NAMES.cpp_header_parsing,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "mac/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        cpp_header_parsing_action = action_config(
            action_name = ACTION_NAMES.cpp_header_parsing,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        cpp_header_parsing_action = action_config(
            action_name = ACTION_NAMES.cpp_header_parsing,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        cpp_header_parsing_action = None

    if (ctx.attr.cpu == "tvos_arm64"):
        assemble_action = action_config(
            action_name = ACTION_NAMES.assemble,
            implies = [
                "objc_arc",
                "no_objc_arc",
                "include_system_dirs",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "unfiltered_cxx_flags",
            ],
            tools = [
                tool(
                    path = "tvos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        assemble_action = action_config(
            action_name = ACTION_NAMES.assemble,
            implies = [
                "objc_arc",
                "no_objc_arc",
                "include_system_dirs",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "unfiltered_cxx_flags",
            ],
            tools = [
                tool(
                    path = "tvsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "x64_windows"):
        assemble_action = action_config(
            action_name = ACTION_NAMES.assemble,
            implies = [
                "objc_arc",
                "no_objc_arc",
                "include_system_dirs",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "<tool_dir>/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7"):
        assemble_action = action_config(
            action_name = ACTION_NAMES.assemble,
            implies = [
                "objc_arc",
                "no_objc_arc",
                "include_system_dirs",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64"):
        assemble_action = action_config(
            action_name = ACTION_NAMES.assemble,
            implies = [
                "objc_arc",
                "no_objc_arc",
                "include_system_dirs",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        assemble_action = action_config(
            action_name = ACTION_NAMES.assemble,
            implies = [
                "objc_arc",
                "no_objc_arc",
                "include_system_dirs",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "mac/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        assemble_action = action_config(
            action_name = ACTION_NAMES.assemble,
            implies = [
                "objc_arc",
                "no_objc_arc",
                "include_system_dirs",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        assemble_action = action_config(
            action_name = ACTION_NAMES.assemble,
            implies = [
                "objc_arc",
                "no_objc_arc",
                "include_system_dirs",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        assemble_action = None

    if (ctx.attr.cpu == "x64_windows"):
        cpp_link_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "<tool_dir>/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7"):
        cpp_link_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "ios/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64"):
        cpp_link_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "iossim/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        cpp_link_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "mac/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        cpp_link_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "tvos/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        cpp_link_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "tvsim/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        cpp_link_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "watchos/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        cpp_link_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "watchsim/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        cpp_link_dynamic_library_action = None

    if (ctx.attr.cpu == "x64_windows"):
        c_compile_action = action_config(
            action_name = ACTION_NAMES.c_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch <architecture>"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "<tool_dir>/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        c_compile_action = action_config(
            action_name = ACTION_NAMES.c_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch arm64"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "unfiltered_cxx_flags",
            ],
            tools = [
                tool(
                    path = "tvos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64"):
        c_compile_action = action_config(
            action_name = ACTION_NAMES.c_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch arm64"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_armv7"):
        c_compile_action = action_config(
            action_name = ACTION_NAMES.c_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch armv7"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        c_compile_action = action_config(
            action_name = ACTION_NAMES.c_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch armv7k"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386"):
        c_compile_action = action_config(
            action_name = ACTION_NAMES.c_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch i386"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        c_compile_action = action_config(
            action_name = ACTION_NAMES.c_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch i386"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        c_compile_action = action_config(
            action_name = ACTION_NAMES.c_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch x86_64"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "unfiltered_cxx_flags",
            ],
            tools = [
                tool(
                    path = "tvsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_x86_64"):
        c_compile_action = action_config(
            action_name = ACTION_NAMES.c_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch x86_64"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        c_compile_action = action_config(
            action_name = ACTION_NAMES.c_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch x86_64"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "mac/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        c_compile_action = None

    if (ctx.attr.cpu == "x64_windows"):
        cpp_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch <architecture>"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "<tool_dir>/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        cpp_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch arm64"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "unfiltered_cxx_flags",
            ],
            tools = [
                tool(
                    path = "tvos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64"):
        cpp_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch arm64"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_armv7"):
        cpp_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch armv7"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        cpp_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch armv7k"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386"):
        cpp_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch i386"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        cpp_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch i386"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        cpp_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch x86_64"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "unfiltered_cxx_flags",
            ],
            tools = [
                tool(
                    path = "tvsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_x86_64"):
        cpp_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch x86_64"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        cpp_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch x86_64"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "mac/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        cpp_compile_action = None

    if (ctx.attr.cpu == "x64_windows"):
        linkstamp_compile_action = action_config(
            action_name = ACTION_NAMES.linkstamp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch <architecture>"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "<tool_dir>/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64"):
        linkstamp_compile_action = action_config(
            action_name = ACTION_NAMES.linkstamp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch arm64"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        linkstamp_compile_action = action_config(
            action_name = ACTION_NAMES.linkstamp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch arm64"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "tvos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_armv7"):
        linkstamp_compile_action = action_config(
            action_name = ACTION_NAMES.linkstamp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch armv7"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        linkstamp_compile_action = action_config(
            action_name = ACTION_NAMES.linkstamp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch armv7k"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386"):
        linkstamp_compile_action = action_config(
            action_name = ACTION_NAMES.linkstamp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch i386"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        linkstamp_compile_action = action_config(
            action_name = ACTION_NAMES.linkstamp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch i386"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_x86_64"):
        linkstamp_compile_action = action_config(
            action_name = ACTION_NAMES.linkstamp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch x86_64"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        linkstamp_compile_action = action_config(
            action_name = ACTION_NAMES.linkstamp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch x86_64"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "mac/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        linkstamp_compile_action = action_config(
            action_name = ACTION_NAMES.linkstamp_compile,
            flag_sets = [
                flag_set(
                    flag_groups = [flag_group(flags = ["-arch x86_64"])],
                ),
            ],
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "tvsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        linkstamp_compile_action = None

    if (ctx.attr.cpu == "tvos_arm64"):
        cpp_module_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_module_compile,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "unfiltered_cxx_flags",
            ],
            tools = [
                tool(
                    path = "tvos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        cpp_module_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_module_compile,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
                "unfiltered_cxx_flags",
            ],
            tools = [
                tool(
                    path = "tvsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "x64_windows"):
        cpp_module_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_module_compile,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "<tool_dir>/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7"):
        cpp_module_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_module_compile,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "ios/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64"):
        cpp_module_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_module_compile,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "iossim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        cpp_module_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_module_compile,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "mac/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        cpp_module_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_module_compile,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchos/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        cpp_module_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_module_compile,
            implies = [
                "preprocessor_defines",
                "include_system_dirs",
                "version_min",
                "objc_arc",
                "no_objc_arc",
                "apple_env",
                "user_compile_flags",
                "sysroot",
                "unfiltered_compile_flags",
                "compiler_input_flags",
                "compiler_output_flags",
            ],
            tools = [
                tool(
                    path = "watchsim/wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        cpp_module_compile_action = None

    if (ctx.attr.cpu == "x64_windows"):
        cpp_link_nodeps_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "<tool_dir>/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7"):
        cpp_link_nodeps_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "ios/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64"):
        cpp_link_nodeps_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "iossim/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        cpp_link_nodeps_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "mac/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        cpp_link_nodeps_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "tvos/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        cpp_link_nodeps_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "tvsim/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        cpp_link_nodeps_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "watchos/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        cpp_link_nodeps_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
            implies = [
                "has_configured_linker_path",
                "symbol_counts",
                "shared_flag",
                "linkstamps",
                "output_execpath_flags",
                "runtime_root_flags",
                "input_param_flags",
                "strip_debug_symbols",
                "linker_param_file",
                "version_min",
                "apple_env",
                "cpp_linker_flags",
                "sysroot",
            ],
            tools = [
                tool(
                    path = "watchsim/clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        cpp_link_nodeps_dynamic_library_action = None

    if (ctx.attr.cpu == "x64_windows"):
        cpp_link_static_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_static_library,
            implies = [
                "runtime_root_flags",
                "archiver_flags",
                "input_param_flags",
                "linker_param_file",
                "apple_env",
            ],
            tools = [
                tool(
                    path = "<tool_dir>/ar_wrapper",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7"):
        cpp_link_static_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_static_library,
            implies = [
                "runtime_root_flags",
                "archiver_flags",
                "input_param_flags",
                "linker_param_file",
                "apple_env",
            ],
            tools = [
                tool(
                    path = "ios/ar_wrapper",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64"):
        cpp_link_static_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_static_library,
            implies = [
                "runtime_root_flags",
                "archiver_flags",
                "input_param_flags",
                "linker_param_file",
                "apple_env",
            ],
            tools = [
                tool(
                    path = "iossim/ar_wrapper",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        cpp_link_static_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_static_library,
            implies = [
                "runtime_root_flags",
                "archiver_flags",
                "input_param_flags",
                "linker_param_file",
                "apple_env",
            ],
            tools = [
                tool(
                    path = "mac/ar_wrapper",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        cpp_link_static_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_static_library,
            implies = [
                "runtime_root_flags",
                "archiver_flags",
                "input_param_flags",
                "linker_param_file",
                "apple_env",
            ],
            tools = [
                tool(
                    path = "tvos/ar_wrapper",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        cpp_link_static_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_static_library,
            implies = [
                "runtime_root_flags",
                "archiver_flags",
                "input_param_flags",
                "linker_param_file",
                "apple_env",
            ],
            tools = [
                tool(
                    path = "tvsim/ar_wrapper",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        cpp_link_static_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_static_library,
            implies = [
                "runtime_root_flags",
                "archiver_flags",
                "input_param_flags",
                "linker_param_file",
                "apple_env",
            ],
            tools = [
                tool(
                    path = "watchos/ar_wrapper",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        cpp_link_static_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_static_library,
            implies = [
                "runtime_root_flags",
                "archiver_flags",
                "input_param_flags",
                "linker_param_file",
                "apple_env",
            ],
            tools = [
                tool(
                    path = "watchsim/ar_wrapper",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        cpp_link_static_library_action = None

    if (ctx.attr.cpu == "x64_windows"):
        objc_fully_link_action = action_config(
            action_name = "objc-fully-link",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-arch_only",
                                "<architecture>",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{fully_linked_archive_path}",
                            ],
                        ),
                        flag_group(
                            flags = ["%{objc_library_exec_paths}"],
                            iterate_over = "objc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{cc_library_exec_paths}"],
                            iterate_over = "cc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{imported_library_exec_paths}"],
                            iterate_over = "imported_library_exec_paths",
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "<tool_dir>/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64"):
        objc_fully_link_action = action_config(
            action_name = "objc-fully-link",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-arch_only",
                                "arm64",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{fully_linked_archive_path}",
                            ],
                        ),
                        flag_group(
                            flags = ["%{objc_library_exec_paths}"],
                            iterate_over = "objc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{cc_library_exec_paths}"],
                            iterate_over = "cc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{imported_library_exec_paths}"],
                            iterate_over = "imported_library_exec_paths",
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "ios/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        objc_fully_link_action = action_config(
            action_name = "objc-fully-link",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-arch_only",
                                "arm64",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{fully_linked_archive_path}",
                            ],
                        ),
                        flag_group(
                            flags = ["%{objc_library_exec_paths}"],
                            iterate_over = "objc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{cc_library_exec_paths}"],
                            iterate_over = "cc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{imported_library_exec_paths}"],
                            iterate_over = "imported_library_exec_paths",
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "tvos/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_armv7"):
        objc_fully_link_action = action_config(
            action_name = "objc-fully-link",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-arch_only",
                                "armv7",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{fully_linked_archive_path}",
                            ],
                        ),
                        flag_group(
                            flags = ["%{objc_library_exec_paths}"],
                            iterate_over = "objc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{cc_library_exec_paths}"],
                            iterate_over = "cc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{imported_library_exec_paths}"],
                            iterate_over = "imported_library_exec_paths",
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "ios/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        objc_fully_link_action = action_config(
            action_name = "objc-fully-link",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-arch_only",
                                "armv7k",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{fully_linked_archive_path}",
                            ],
                        ),
                        flag_group(
                            flags = ["%{objc_library_exec_paths}"],
                            iterate_over = "objc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{cc_library_exec_paths}"],
                            iterate_over = "cc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{imported_library_exec_paths}"],
                            iterate_over = "imported_library_exec_paths",
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "watchos/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386"):
        objc_fully_link_action = action_config(
            action_name = "objc-fully-link",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-arch_only",
                                "i386",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{fully_linked_archive_path}",
                            ],
                        ),
                        flag_group(
                            flags = ["%{objc_library_exec_paths}"],
                            iterate_over = "objc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{cc_library_exec_paths}"],
                            iterate_over = "cc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{imported_library_exec_paths}"],
                            iterate_over = "imported_library_exec_paths",
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "iossim/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        objc_fully_link_action = action_config(
            action_name = "objc-fully-link",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-arch_only",
                                "i386",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{fully_linked_archive_path}",
                            ],
                        ),
                        flag_group(
                            flags = ["%{objc_library_exec_paths}"],
                            iterate_over = "objc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{cc_library_exec_paths}"],
                            iterate_over = "cc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{imported_library_exec_paths}"],
                            iterate_over = "imported_library_exec_paths",
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "watchsim/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_x86_64"):
        objc_fully_link_action = action_config(
            action_name = "objc-fully-link",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-arch_only",
                                "x86_64",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{fully_linked_archive_path}",
                            ],
                        ),
                        flag_group(
                            flags = ["%{objc_library_exec_paths}"],
                            iterate_over = "objc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{cc_library_exec_paths}"],
                            iterate_over = "cc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{imported_library_exec_paths}"],
                            iterate_over = "imported_library_exec_paths",
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "iossim/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        objc_fully_link_action = action_config(
            action_name = "objc-fully-link",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-arch_only",
                                "x86_64",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{fully_linked_archive_path}",
                            ],
                        ),
                        flag_group(
                            flags = ["%{objc_library_exec_paths}"],
                            iterate_over = "objc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{cc_library_exec_paths}"],
                            iterate_over = "cc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{imported_library_exec_paths}"],
                            iterate_over = "imported_library_exec_paths",
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "mac/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        objc_fully_link_action = action_config(
            action_name = "objc-fully-link",
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-static",
                                "-arch_only",
                                "x86_64",
                                "-syslibroot",
                                "%{sdk_dir}",
                                "-o",
                                "%{fully_linked_archive_path}",
                            ],
                        ),
                        flag_group(
                            flags = ["%{objc_library_exec_paths}"],
                            iterate_over = "objc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{cc_library_exec_paths}"],
                            iterate_over = "cc_library_exec_paths",
                        ),
                        flag_group(
                            flags = ["%{imported_library_exec_paths}"],
                            iterate_over = "imported_library_exec_paths",
                        ),
                    ],
                ),
            ],
            implies = ["apple_env"],
            tools = [
                tool(
                    path = "tvsim/libtool",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    else:
        objc_fully_link_action = None

    if (ctx.attr.cpu == "x64_windows"):
        strip_action = action_config(
            action_name = ACTION_NAMES.strip,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-S", "-o", "%{output_file}"]),
                        flag_group(
                            flags = ["%{stripopts}"],
                            iterate_over = "stripopts",
                        ),
                        flag_group(flags = ["%{input_file}"]),
                    ],
                ),
            ],
            tools = [tool(path = "<tool_dir>/strip")],
        )
    elif (ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7"):
        strip_action = action_config(
            action_name = ACTION_NAMES.strip,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-S", "-o", "%{output_file}"]),
                        flag_group(
                            flags = ["%{stripopts}"],
                            iterate_over = "stripopts",
                        ),
                        flag_group(flags = ["%{input_file}"]),
                    ],
                ),
            ],
            tools = [tool(path = "ios/strip")],
        )
    elif (ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64"):
        strip_action = action_config(
            action_name = ACTION_NAMES.strip,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-S", "-o", "%{output_file}"]),
                        flag_group(
                            flags = ["%{stripopts}"],
                            iterate_over = "stripopts",
                        ),
                        flag_group(flags = ["%{input_file}"]),
                    ],
                ),
            ],
            tools = [tool(path = "iossim/strip")],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        strip_action = action_config(
            action_name = ACTION_NAMES.strip,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-S", "-o", "%{output_file}"]),
                        flag_group(
                            flags = ["%{stripopts}"],
                            iterate_over = "stripopts",
                        ),
                        flag_group(flags = ["%{input_file}"]),
                    ],
                ),
            ],
            tools = [tool(path = "mac/strip")],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        strip_action = action_config(
            action_name = ACTION_NAMES.strip,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-S", "-o", "%{output_file}"]),
                        flag_group(
                            flags = ["%{stripopts}"],
                            iterate_over = "stripopts",
                        ),
                        flag_group(flags = ["%{input_file}"]),
                    ],
                ),
            ],
            tools = [tool(path = "tvos/strip")],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        strip_action = action_config(
            action_name = ACTION_NAMES.strip,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-S", "-o", "%{output_file}"]),
                        flag_group(
                            flags = ["%{stripopts}"],
                            iterate_over = "stripopts",
                        ),
                        flag_group(flags = ["%{input_file}"]),
                    ],
                ),
            ],
            tools = [tool(path = "tvsim/strip")],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        strip_action = action_config(
            action_name = ACTION_NAMES.strip,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-S", "-o", "%{output_file}"]),
                        flag_group(
                            flags = ["%{stripopts}"],
                            iterate_over = "stripopts",
                        ),
                        flag_group(flags = ["%{input_file}"]),
                    ],
                ),
            ],
            tools = [tool(path = "watchos/strip")],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        strip_action = action_config(
            action_name = ACTION_NAMES.strip,
            flag_sets = [
                flag_set(
                    flag_groups = [
                        flag_group(flags = ["-S", "-o", "%{output_file}"]),
                        flag_group(
                            flags = ["%{stripopts}"],
                            iterate_over = "stripopts",
                        ),
                        flag_group(flags = ["%{input_file}"]),
                    ],
                ),
            ],
            tools = [tool(path = "watchsim/strip")],
        )
    else:
        strip_action = None

    action_configs = [
        strip_action,
        c_compile_action,
        cpp_compile_action,
        linkstamp_compile_action,
        cpp_module_compile_action,
        cpp_header_parsing_action,
        objc_compile_action,
        objcpp_compile_action,
        assemble_action,
        preprocess_assemble_action,
        objc_archive_action,
        objc_executable_action,
        objcpp_executable_action,
        cpp_link_executable_action,
        cpp_link_dynamic_library_action,
        cpp_link_nodeps_dynamic_library_action,
        cpp_link_static_library_action,
        objc_fully_link_action,
    ]

    action_configs.extend([_action_name_to_action[name] for name in ctx.attr.action_configs])

    has_configured_linker_path_feature = feature(name = "has_configured_linker_path")

    use_objc_modules_feature = feature(
        name = "use_objc_modules",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.objc_compile, ACTION_NAMES.objcpp_compile],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-fmodule-name=%{module_name}",
                            "-iquote",
                            "%{module_maps_dir}",
                            "-fmodules-cache-path=%{modules_cache_path}",
                        ],
                    ),
                ],
            ),
        ],
    )

    if (ctx.attr.cpu == "ios_arm64" or
        ctx.attr.cpu == "ios_armv7" or
        ctx.attr.cpu == "ios_i386" or
        ctx.attr.cpu == "ios_x86_64" or
        ctx.attr.cpu == "tvos_arm64" or
        ctx.attr.cpu == "tvos_x86_64" or
        ctx.attr.cpu == "watchos_armv7k" or
        ctx.attr.cpu == "watchos_i386" or
        ctx.attr.cpu == "x64_windows"):
        apply_implicit_frameworks_feature = feature(
            name = "apply_implicit_frameworks",
            flag_sets = [
                flag_set(
                    actions = ["objc-executable", "objc++-executable"],
                    flag_groups = [
                        flag_group(
                            flags = ["-framework Foundation", "-framework UIKit"],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        apply_implicit_frameworks_feature = feature(
            name = "apply_implicit_frameworks",
            flag_sets = [
                flag_set(
                    actions = ["objc-executable", "objc++-executable"],
                    flag_groups = [flag_group(flags = ["-framework Foundation"])],
                ),
            ],
        )
    else:
        apply_implicit_frameworks_feature = None

    if (ctx.attr.cpu == "ios_arm64"):
        unfiltered_compile_flags_feature = feature(
            name = "unfiltered_compile_flags",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-no-canonical-prefixes",
                                "-pthread",
                                "-target",
                                "arm64-apple-ios",
                            ],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        unfiltered_compile_flags_feature = feature(
            name = "unfiltered_compile_flags",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-no-canonical-prefixes",
                                "-pthread",
                                "-target",
                                "arm64-apple-tvos",
                            ],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_armv7"):
        unfiltered_compile_flags_feature = feature(
            name = "unfiltered_compile_flags",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-no-canonical-prefixes",
                                "-pthread",
                                "-target",
                                "armv7-apple-ios",
                            ],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        unfiltered_compile_flags_feature = feature(
            name = "unfiltered_compile_flags",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-no-canonical-prefixes",
                                "-pthread",
                                "-target",
                                "armv7k-apple-watchos",
                            ],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386"):
        unfiltered_compile_flags_feature = feature(
            name = "unfiltered_compile_flags",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-no-canonical-prefixes",
                                "-pthread",
                                "-target",
                                "i386-apple-ios",
                            ],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        unfiltered_compile_flags_feature = feature(
            name = "unfiltered_compile_flags",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-no-canonical-prefixes",
                                "-pthread",
                                "-target",
                                "i386-apple-watchos",
                            ],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_x86_64"):
        unfiltered_compile_flags_feature = feature(
            name = "unfiltered_compile_flags",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-no-canonical-prefixes",
                                "-pthread",
                                "-target",
                                "x86_64-apple-ios",
                            ],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        unfiltered_compile_flags_feature = feature(
            name = "unfiltered_compile_flags",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-no-canonical-prefixes",
                                "-pthread",
                                "-target",
                                "x86_64-apple-tvos",
                            ],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64" or
          ctx.attr.cpu == "x64_windows"):
        unfiltered_compile_flags_feature = feature(
            name = "unfiltered_compile_flags",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                    ],
                    flag_groups = [
                        flag_group(flags = ["-no-canonical-prefixes", "-pthread"]),
                    ],
                ),
            ],
        )
    else:
        unfiltered_compile_flags_feature = None

    fastbuild_feature = feature(
        name = "fastbuild",
        implies = ["fastbuild_only_flag"],
    )

    archiver_flags_feature = feature(
        name = "archiver_flags",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["rcs", "%{output_execpath}"],
                        expand_if_available = "output_execpath",
                    ),
                ],
            ),
        ],
    )

    module_maps_feature = feature(name = "module_maps", enabled = True)

    dependency_file_feature = feature(
        name = "dependency_file",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-MD", "-MF", "%{dependency_file}"],
                        expand_if_available = "dependency_file",
                    ),
                ],
            ),
        ],
    )

    opt_only_flag_feature = feature(
        name = "opt_only_flag",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.objc_compile],
                flag_groups = [flag_group(flags = ["--OPT_ONLY_FLAG"])],
            ),
        ],
    )

    if (ctx.attr.cpu == "x64_windows"):
        version_min_feature = feature(
            name = "version_min",
            flag_sets = [
                flag_set(
                    actions = [
                        "objc-executable",
                        "objc++-executable",
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = ["-m<platform_for_version_min>-version-min=%{version_min}"],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64"):
        version_min_feature = feature(
            name = "version_min",
            flag_sets = [
                flag_set(
                    actions = [
                        "objc-executable",
                        "objc++-executable",
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = ["-mios-simulator-version-min=%{version_min}"],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7"):
        version_min_feature = feature(
            name = "version_min",
            flag_sets = [
                flag_set(
                    actions = [
                        "objc-executable",
                        "objc++-executable",
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = ["-miphoneos-version-min=%{version_min}"],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        version_min_feature = feature(
            name = "version_min",
            flag_sets = [
                flag_set(
                    actions = [
                        "objc-executable",
                        "objc++-executable",
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = ["-mtvos-simulator-version-min=%{version_min}"],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_i386"):
        version_min_feature = feature(
            name = "version_min",
            flag_sets = [
                flag_set(
                    actions = [
                        "objc-executable",
                        "objc++-executable",
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = ["-mwatchos-simulator-version-min=%{version_min}"],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        version_min_feature = feature(
            name = "version_min",
            flag_sets = [
                flag_set(
                    actions = [
                        "objc-executable",
                        "objc++-executable",
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = ["-mwatchos-version-min=%{version_min}"],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        version_min_feature = feature(
            name = "version_min",
            flag_sets = [
                flag_set(
                    actions = [
                        "objc-executable",
                        "objc++-executable",
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [
                        flag_group(flags = ["-mmacosx-version-min=%{version_min}"]),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64"):
        version_min_feature = feature(
            name = "version_min",
            flag_sets = [
                flag_set(
                    actions = [
                        "objc-executable",
                        "objc++-executable",
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [flag_group(flags = ["-mtvos-version-min=%{version_min}"])],
                ),
            ],
        )
    else:
        version_min_feature = None

    asan_feature = feature(
        name = "asan",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-O1",
                            "-gmlt",
                            "-fsanitize=address,bool,float-cast-overflow,integer-divide-by-zero,return,returns-nonnull-attribute,shift-exponent,unreachable,vla-bound",
                            "-fno-sanitize-recover=all",
                            "-DHEAPCHECK_DISABLE",
                            "-DADDRESS_SANITIZER",
                            "-D_GLIBCXX_ADDRESS_SANITIZER_ANNOTATIONS",
                            "-fno-omit-frame-pointer",
                        ],
                    ),
                ],
            ),
            flag_set(
                actions = _ALL_LINK_ACTIONS,
                flag_groups = [
                    flag_group(
                        flags = [
                            "-fsanitize=address,bool,float-cast-overflow,integer-divide-by-zero,return,returns-nonnull-attribute,shift-exponent,unreachable,vla-bound",
                            "-fsanitize-link-c++-runtime",
                        ],
                    ),
                ],
            ),
        ],
    )

    llvm_coverage_map_format_feature = feature(
        name = "llvm_coverage_map_format",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-fprofile-instr-generate", "-fcoverage-mapping", "-g"],
                    ),
                ],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                    "objc-executable",
                    "objc++-executable",
                ],
                flag_groups = [flag_group(flags = ["-fprofile-instr-generate"])],
            ),
        ],
        requires = [feature_set(features = ["coverage"])],
        provides = ["profile"],
    )

    compiler_input_flags_feature = feature(
        name = "compiler_input_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-c", "%{source_file}"],
                        expand_if_available = "source_file",
                    ),
                ],
            ),
        ],
    )

    linkstamps_feature = feature(
        name = "linkstamps",
        flag_sets = [
            flag_set(
                actions = _NON_OBJC_LINK_ACTIONS,
                flag_groups = [
                    flag_group(
                        flags = ["%{linkstamp_paths}"],
                        iterate_over = "linkstamp_paths",
                        expand_if_available = "linkstamp_paths",
                    ),
                ],
            ),
        ],
    )

    if (ctx.attr.cpu == "ios_arm64" or
        ctx.attr.cpu == "ios_armv7" or
        ctx.attr.cpu == "tvos_arm64" or
        ctx.attr.cpu == "watchos_armv7k"):
        bitcode_embedded_feature = feature(
            name = "bitcode_embedded",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                        "objc-executable",
                        "objc++-executable",
                    ],
                    flag_groups = [flag_group(flags = ["-fembed-bitcode"])],
                ),
                flag_set(
                    actions = ["objc-executable", "objc++-executable"],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-Xlinker",
                                "-bitcode_verify",
                                "-Xlinker",
                                "-bitcode_hide_symbols",
                                "-Xlinker",
                                "-bitcode_symbol_map",
                                "-Xlinker",
                                "BITCODE_TOUCH_SYMBOL_MAP=%{bitcode_symbol_map_path}",
                            ],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64" or
          ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64" or
          ctx.attr.cpu == "tvos_x86_64" or
          ctx.attr.cpu == "watchos_i386" or
          ctx.attr.cpu == "x64_windows"):
        bitcode_embedded_feature = feature(name = "bitcode_embedded")
    else:
        bitcode_embedded_feature = None

    linker_param_file_feature = feature(
        name = "linker_param_file",
        flag_sets = [
            flag_set(
                actions = _NON_OBJC_LINK_ACTIONS,
                flag_groups = [
                    flag_group(
                        flags = ["-Wl,@%{linker_param_file}"],
                        expand_if_available = "linker_param_file",
                    ),
                ],
            ),
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["@%{linker_param_file}"],
                        expand_if_available = "linker_param_file",
                    ),
                ],
            ),
        ],
    )

    shared_flag_feature = feature(
        name = "shared_flag",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(flags = ["-shared"])],
            ),
        ],
    )

    output_execpath_flags_feature = feature(
        name = "output_execpath_flags",
        flag_sets = [
            flag_set(
                actions = _NON_OBJC_LINK_ACTIONS,
                flag_groups = [
                    flag_group(
                        flags = ["-o", "%{output_execpath}"],
                        expand_if_available = "output_execpath",
                    ),
                ],
            ),
        ],
    )

    if (ctx.attr.cpu == "tvos_arm64"):
        cpp_linker_flags_feature = feature(
            name = "cpp_linker_flags",
            flag_sets = [
                flag_set(
                    actions = _NON_OBJC_LINK_ACTIONS,
                    flag_groups = [
                        flag_group(
                            flags = ["-lc++", "-target", "arm64-apple-tvos"],
                        ),
                    ],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_dynamic_library,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = ["-lc++", "-target", "arm64-apple-tvos"],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        cpp_linker_flags_feature = feature(
            name = "cpp_linker_flags",
            flag_sets = [
                flag_set(
                    actions = _NON_OBJC_LINK_ACTIONS,
                    flag_groups = [
                        flag_group(
                            flags = ["-lc++", "-target", "x86_64-apple-tvos"],
                        ),
                    ],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_dynamic_library,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = ["-lc++", "-target", "x86_64-apple-tvos"],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64" or
          ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7" or
          ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64" or
          ctx.attr.cpu == "watchos_armv7k" or
          ctx.attr.cpu == "watchos_i386" or
          ctx.attr.cpu == "x64_windows"):
        cpp_linker_flags_feature = feature(
            name = "cpp_linker_flags",
            flag_sets = [
                flag_set(
                    actions = _NON_OBJC_LINK_ACTIONS,
                    flag_groups = [],
                ),
            ],
        )
    else:
        cpp_linker_flags_feature = None

    sysroot_feature = feature(
        name = "sysroot",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["--sysroot=%{sysroot}"],
                        expand_if_available = "sysroot",
                    ),
                ],
            ),
        ],
    )

    fdo_optimize_feature = feature(
        name = "fdo_optimize",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-fprofile-use=%{fdo_profile_path}",
                            "-Xclang-only=-Wno-profile-instr-unprofiled",
                            "-Xclang-only=-Wno-profile-instr-out-of-date",
                            "-fprofile-correction",
                        ],
                        expand_if_available = "fdo_profile_path",
                    ),
                ],
            ),
        ],
        provides = ["profile"],
    )

    coverage_feature = feature(name = "coverage")

    if (ctx.attr.cpu == "x64_windows"):
        default_compile_flags_feature = feature(
            name = "default_compile_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-DCOMPILER_GCC3",
                                "-DCOMPILER_GCC4",
                                "-Dunix",
                                "-DOS_IOS",
                                "-DU_HAVE_NL_LANGINFO_CODESET=0",
                                "-DU_HAVE_STD_STRING",
                                "-D__STDC_FORMAT_MACROS",
                                "-fcolor-diagnostics",
                            ],
                        ),
                    ],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [flag_group(flags = ["-O0", "-DDEBUG"])],
                    with_features = [with_feature_set(features = ["fastbuild"])],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = ["-Os", "-DNDEBUG", "-DNS_BLOCK_ASSERTIONS=1"],
                        ),
                    ],
                    with_features = [with_feature_set(features = ["opt"])],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [flag_group(flags = ["-g", "-DDEBUG"])],
                    with_features = [with_feature_set(features = ["dbg"])],
                ),
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
                    flag_groups = [flag_group(flags = ["-std=gnu++11", "-stdlib=libc++"])],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7" or
          ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64" or
          ctx.attr.cpu == "watchos_armv7k" or
          ctx.attr.cpu == "watchos_i386"):
        default_compile_flags_feature = feature(
            name = "default_compile_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-DCOMPILER_GCC3",
                                "-DCOMPILER_GCC4",
                                "-Dunix",
                                "-DOS_IOS",
                                "-DU_HAVE_NL_LANGINFO_CODESET=0",
                                "-DU_HAVE_STD_STRING",
                                "-D__STDC_FORMAT_MACROS",
                                "-fcolor-diagnostics",
                            ],
                        ),
                    ],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [flag_group(flags = ["-O0", "-DDEBUG"])],
                    with_features = [with_feature_set(features = ["fastbuild"])],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [flag_group(flags = ["-Os", "-DNDEBUG"])],
                    with_features = [with_feature_set(features = ["opt"])],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [flag_group(flags = ["-g", "-DDEBUG"])],
                    with_features = [with_feature_set(features = ["dbg"])],
                ),
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
                    flag_groups = [flag_group(flags = ["-std=gnu++11", "-stdlib=libc++"])],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        default_compile_flags_feature = feature(
            name = "default_compile_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-DCOMPILER_GCC3",
                                "-DCOMPILER_GCC4",
                                "-Dunix",
                                "-DOS_MACOSX",
                                "-DU_HAVE_NL_LANGINFO_CODESET=0",
                                "-DU_HAVE_STD_STRING",
                                "-D__STDC_FORMAT_MACROS",
                                "-fcolor-diagnostics",
                            ],
                        ),
                    ],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [flag_group(flags = ["-O0", "-DDEBUG"])],
                    with_features = [with_feature_set(features = ["fastbuild"])],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [flag_group(flags = ["-Os", "-DNDEBUG"])],
                    with_features = [with_feature_set(features = ["opt"])],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [flag_group(flags = ["-g", "-DDEBUG"])],
                    with_features = [with_feature_set(features = ["dbg"])],
                ),
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
                    flag_groups = [flag_group(flags = ["-std=gnu++11", "-stdlib=libc++"])],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64" or
          ctx.attr.cpu == "tvos_x86_64"):
        default_compile_flags_feature = feature(
            name = "default_compile_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-DCOMPILER_GCC3",
                                "-DCOMPILER_GCC4",
                                "-Dunix",
                                "-DOS_TVOS",
                                "-DU_HAVE_NL_LANGINFO_CODESET=0",
                                "-DU_HAVE_STD_STRING",
                                "-D__STDC_FORMAT_MACROS",
                                "-fcolor-diagnostics",
                            ],
                        ),
                    ],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [flag_group(flags = ["-O0", "-DDEBUG"])],
                    with_features = [with_feature_set(features = ["fastbuild"])],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = ["-Os", "-DNDEBUG", "-DNS_BLOCK_ASSERTIONS=1"],
                        ),
                    ],
                    with_features = [with_feature_set(features = ["opt"])],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                        ACTION_NAMES.linkstamp_compile,
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_module_codegen,
                        ACTION_NAMES.lto_backend,
                        ACTION_NAMES.clif_match,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [flag_group(flags = ["-g", "-DDEBUG"])],
                    with_features = [with_feature_set(features = ["dbg"])],
                ),
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
                    flag_groups = [flag_group(flags = ["-std=gnu++11", "-stdlib=libc++"])],
                ),
            ],
        )
    else:
        default_compile_flags_feature = None

    generate_linkmap_feature = feature(
        name = "generate_linkmap",
        flag_sets = [
            flag_set(
                actions = ["objc-executable", "objc++-executable"],
                flag_groups = [
                    flag_group(
                        flags = ["-Xlinker -map", "-Xlinker %{linkmap_exec_path}"],
                    ),
                ],
            ),
        ],
    )

    input_param_flags_feature = feature(
        name = "input_param_flags",
        flag_sets = [
            flag_set(
                actions = _NON_OBJC_LINK_ACTIONS +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["-L%{library_search_directories}"],
                        iterate_over = "library_search_directories",
                        expand_if_available = "library_search_directories",
                    ),
                ],
            ),
            flag_set(
                actions = _NON_OBJC_LINK_ACTIONS +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["%{libopts}"],
                        iterate_over = "libopts",
                        expand_if_available = "libopts",
                    ),
                ],
            ),
            flag_set(
                actions = _NON_OBJC_LINK_ACTIONS +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["-Wl,-force_load,%{whole_archive_linker_params}"],
                        iterate_over = "whole_archive_linker_params",
                        expand_if_available = "whole_archive_linker_params",
                    ),
                ],
            ),
            flag_set(
                actions = _NON_OBJC_LINK_ACTIONS +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["%{linker_input_params}"],
                        iterate_over = "linker_input_params",
                        expand_if_available = "linker_input_params",
                    ),
                ],
            ),
            flag_set(
                actions = _NON_OBJC_LINK_ACTIONS +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        iterate_over = "libraries_to_link",
                        flag_groups = [
                            flag_group(
                                flags = ["-Wl,--start-lib"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file_group",
                                ),
                            ),
                            flag_group(
                                iterate_over = "libraries_to_link.object_files",
                                flag_groups = [
                                    flag_group(
                                        flags = ["%{libraries_to_link.object_files}"],
                                        expand_if_false = "libraries_to_link.is_whole_archive",
                                    ),
                                    flag_group(
                                        flags = ["-Wl,-force_load,%{libraries_to_link.object_files}"],
                                        expand_if_true = "libraries_to_link.is_whole_archive",
                                    ),
                                ],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file_group",
                                ),
                            ),
                            flag_group(
                                flags = ["-Wl,--end-lib"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file_group",
                                ),
                            ),
                            flag_group(
                                flag_groups = [
                                    flag_group(
                                        flags = ["%{libraries_to_link.name}"],
                                        expand_if_false = "libraries_to_link.is_whole_archive",
                                    ),
                                    flag_group(
                                        flags = ["-Wl,-force_load,%{libraries_to_link.name}"],
                                        expand_if_true = "libraries_to_link.is_whole_archive",
                                    ),
                                ],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file",
                                ),
                            ),
                            flag_group(
                                flag_groups = [
                                    flag_group(
                                        flags = ["%{libraries_to_link.name}"],
                                        expand_if_false = "libraries_to_link.is_whole_archive",
                                    ),
                                    flag_group(
                                        flags = ["-Wl,-force_load,%{libraries_to_link.name}"],
                                        expand_if_true = "libraries_to_link.is_whole_archive",
                                    ),
                                ],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "interface_library",
                                ),
                            ),
                            flag_group(
                                flag_groups = [
                                    flag_group(
                                        flags = ["%{libraries_to_link.name}"],
                                        expand_if_false = "libraries_to_link.is_whole_archive",
                                    ),
                                    flag_group(
                                        flags = ["-Wl,-force_load,%{libraries_to_link.name}"],
                                        expand_if_true = "libraries_to_link.is_whole_archive",
                                    ),
                                ],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "static_library",
                                ),
                            ),
                            flag_group(
                                flag_groups = [
                                    flag_group(
                                        flags = ["-l%{libraries_to_link.name}"],
                                        expand_if_false = "libraries_to_link.is_whole_archive",
                                    ),
                                    flag_group(
                                        flags = ["-Wl,-force_load,-l%{libraries_to_link.name}"],
                                        expand_if_true = "libraries_to_link.is_whole_archive",
                                    ),
                                ],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "dynamic_library",
                                ),
                            ),
                            flag_group(
                                flag_groups = [
                                    flag_group(
                                        flags = ["-l:%{libraries_to_link.name}"],
                                        expand_if_false = "libraries_to_link.is_whole_archive",
                                    ),
                                    flag_group(
                                        flags = ["-Wl,-force_load,-l:%{libraries_to_link.name}"],
                                        expand_if_true = "libraries_to_link.is_whole_archive",
                                    ),
                                ],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "versioned_dynamic_library",
                                ),
                            ),
                        ],
                        expand_if_available = "libraries_to_link",
                    ),
                ],
            ),
        ],
    )

    xcode_5_0_feature = feature(
        name = "xcode_5.0",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    "c++-header-preprocessing",
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-DXCODE_FEATURE_FOR_TESTING=xcode_5.0"],
                    ),
                ],
            ),
        ],
    )

    user_link_flags_feature = feature(
        name = "user_link_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = _ALL_LINK_ACTIONS,
                flag_groups = [
                    flag_group(
                        flags = ["%{user_link_flags}"],
                        iterate_over = "user_link_flags",
                        expand_if_available = "user_link_flags",
                    ),
                ],
            ),
        ],
    )

    pic_feature = feature(
        name = "pic",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [
                    flag_group(flags = ["-fPIC"], expand_if_available = "pic"),
                ],
            ),
        ],
    )

    generate_dsym_file_feature = feature(
        name = "generate_dsym_file",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                    "objc-executable",
                    "objc++-executable",
                ],
                flag_groups = [flag_group(flags = ["-g"])],
            ),
            flag_set(
                actions = ["objc-executable", "objc++-executable"],
                flag_groups = [
                    flag_group(
                        flags = [
                            "DSYM_HINT_LINKED_BINARY=%{linked_binary}",
                            "DSYM_HINT_DSYM_PATH=%{dsym_path}",
                        ],
                    ),
                ],
            ),
        ],
    )

    no_enable_modules_feature = feature(
        name = "no_enable_modules",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.objc_compile, ACTION_NAMES.objcpp_compile],
                flag_groups = [flag_group(flags = ["-fmodule-maps"])],
            ),
        ],
        requires = [feature_set(features = ["use_objc_modules"])],
    )

    autofdo_feature = feature(
        name = "autofdo",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-fauto-profile=%{fdo_profile_path}",
                            "-fprofile-correction",
                        ],
                        expand_if_available = "fdo_profile_path",
                    ),
                ],
            ),
        ],
        provides = ["profile"],
    )

    if (ctx.attr.cpu == "darwin_x86_64"):
        link_cocoa_feature = feature(
            name = "link_cocoa",
            flag_sets = [
                flag_set(
                    actions = _OBJC_LINK_ACTIONS,
                    flag_groups = [flag_group(flags = ["-framework Cocoa"])],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7" or
          ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64" or
          ctx.attr.cpu == "tvos_arm64" or
          ctx.attr.cpu == "tvos_x86_64" or
          ctx.attr.cpu == "watchos_armv7k" or
          ctx.attr.cpu == "watchos_i386" or
          ctx.attr.cpu == "x64_windows"):
        link_cocoa_feature = feature(name = "link_cocoa")
    else:
        link_cocoa_feature = None

    objc_actions_feature = feature(
        name = "objc_actions",
        implies = [
            "objc-compile",
            "objc++-compile",
            "objc-fully-link",
            "objc-archive",
            "objc-executable",
            "objc++-executable",
            "assemble",
            "preprocess-assemble",
            "c-compile",
            "c++-compile",
            "c++-link-static-library",
            "c++-link-dynamic-library",
            "c++-link-nodeps-dynamic-library",
            "c++-link-executable",
        ],
    )

    objc_arc_feature = feature(
        name = "objc_arc",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-fobjc-arc"],
                        expand_if_available = "objc_arc",
                    ),
                ],
            ),
        ],
    )

    apple_env_feature = feature(
        name = "apple_env",
        env_sets = [
            env_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                    "objc-archive",
                    "objc-fully-link",
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_static_library,
                    "objc-executable",
                    "objc++-executable",
                ],
                env_entries = [
                    env_entry(
                        key = "XCODE_VERSION_OVERRIDE",
                        value = "%{xcode_version_override_value}",
                    ),
                    env_entry(
                        key = "APPLE_SDK_VERSION_OVERRIDE",
                        value = "%{apple_sdk_version_override_value}",
                    ),
                    env_entry(
                        key = "APPLE_SDK_PLATFORM",
                        value = "%{apple_sdk_platform_value}",
                    ),
                ],
            ),
        ],
    )

    user_compile_flags_feature = feature(
        name = "user_compile_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["%{user_compile_flags}"],
                        iterate_over = "user_compile_flags",
                        expand_if_available = "user_compile_flags",
                    ),
                ],
            ),
        ],
    )

    if (ctx.attr.cpu == "tvos_arm64"):
        default_link_flags_feature = feature(
            name = "default_link_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = _ALL_LINK_ACTIONS,
                    flag_groups = [
                        flag_group(
                            flags = ["-lc++", "-target", "arm64-apple-tvos"],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "watchos_armv7k"):
        default_link_flags_feature = feature(
            name = "default_link_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = _ALL_LINK_ACTIONS,
                    flag_groups = [
                        flag_group(
                            flags = ["-lc++", "-target", "armv7-apple-watchos"],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_x86_64"):
        default_link_flags_feature = feature(
            name = "default_link_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = _ALL_LINK_ACTIONS,
                    flag_groups = [
                        flag_group(
                            flags = ["-lc++", "-target", "x86_64-apple-ios"],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_x86_64"):
        default_link_flags_feature = feature(
            name = "default_link_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = _ALL_LINK_ACTIONS,
                    flag_groups = [
                        flag_group(
                            flags = ["-lc++", "-target", "x86_64-apple-tvos"],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_arm64"):
        default_link_flags_feature = feature(
            name = "default_link_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = _ALL_LINK_ACTIONS,
                    flag_groups = [
                        flag_group(flags = ["-lc++", "-target", "arm64-apple-ios"]),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_armv7"):
        default_link_flags_feature = feature(
            name = "default_link_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = _ALL_LINK_ACTIONS,
                    flag_groups = [
                        flag_group(flags = ["-lc++", "-target", "armv7-apple-ios"]),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "watchos_i386"):
        default_link_flags_feature = feature(
            name = "default_link_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = _ALL_LINK_ACTIONS,
                    flag_groups = [
                        flag_group(flags = ["-lc++", "-target", "i386-apple-ios"]),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64" or
          ctx.attr.cpu == "x64_windows"):
        default_link_flags_feature = feature(
            name = "default_link_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = _ALL_LINK_ACTIONS,
                    flag_groups = [flag_group(flags = ["-lc++"])],
                ),
            ],
        )
    else:
        default_link_flags_feature = None

    only_doth_headers_in_module_maps_feature = feature(name = "only_doth_headers_in_module_maps")

    runtime_root_flags_feature = feature(
        name = "runtime_root_flags",
        flag_sets = [
            flag_set(
                actions = _NON_OBJC_LINK_ACTIONS +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-Wl,-rpath,$ORIGIN/%{runtime_library_search_directories}",
                        ],
                        iterate_over = "runtime_library_search_directories",
                        expand_if_available = "runtime_library_search_directories",
                    ),
                ],
            ),
            flag_set(
                actions = _NON_OBJC_LINK_ACTIONS +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["%{runtime_root_flags}"],
                        iterate_over = "runtime_root_flags",
                        expand_if_available = "runtime_root_flags",
                    ),
                ],
            ),
            flag_set(
                actions = _NON_OBJC_LINK_ACTIONS +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["%{runtime_root_entries}"],
                        iterate_over = "runtime_root_entries",
                        expand_if_available = "runtime_root_entries",
                    ),
                ],
            ),
        ],
    )

    xcode_7_3_feature = feature(
        name = "xcode_7.3",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    "c++-header-preprocessing",
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-DXCODE_FEATURE_FOR_TESTING=xcode_7.3"],
                    ),
                ],
            ),
        ],
    )

    include_paths_feature = feature(
        name = "include_paths",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.clif_match,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-iquote", "%{quote_include_paths}"],
                        iterate_over = "quote_include_paths",
                    ),
                    flag_group(
                        flags = ["-I%{include_paths}"],
                        iterate_over = "include_paths",
                    ),
                    flag_group(
                        flags = ["-isystem", "%{system_include_paths}"],
                        iterate_over = "system_include_paths",
                    ),
                ],
            ),
        ],
    )

    if (ctx.attr.cpu == "tvos_arm64" or
        ctx.attr.cpu == "tvos_x86_64"):
        unfiltered_cxx_flags_feature = feature(
            name = "unfiltered_cxx_flags",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                    ],
                    flag_groups = [
                        flag_group(flags = ["-no-canonical-prefixes", "-pthread"]),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64" or
          ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7" or
          ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64" or
          ctx.attr.cpu == "watchos_armv7k" or
          ctx.attr.cpu == "watchos_i386" or
          ctx.attr.cpu == "x64_windows"):
        unfiltered_cxx_flags_feature = feature(name = "unfiltered_cxx_flags")
    else:
        unfiltered_cxx_flags_feature = None

    no_legacy_features_feature = feature(name = "no_legacy_features")

    strip_debug_symbols_feature = feature(
        name = "strip_debug_symbols",
        flag_sets = [
            flag_set(
                actions = _NON_OBJC_LINK_ACTIONS,
                flag_groups = [
                    flag_group(
                        flags = ["-Wl,-S"],
                        expand_if_available = "strip_debug_symbols",
                    ),
                ],
            ),
        ],
    )

    force_pic_flags_feature = feature(
        name = "force_pic_flags",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_executable],
                flag_groups = [
                    flag_group(
                        flags = ["-Wl,-pie"],
                        expand_if_available = "force_pic",
                    ),
                ],
            ),
        ],
    )

    pch_feature = feature(
        name = "pch",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [flag_group(flags = ["-include", "%{pch_file}"])],
            ),
        ],
    )

    no_objc_arc_feature = feature(
        name = "no_objc_arc",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-fno-objc-arc"],
                        expand_if_available = "no_objc_arc",
                    ),
                ],
            ),
        ],
    )

    includes_feature = feature(
        name = "includes",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-include", "%{includes}"],
                        iterate_over = "includes",
                        expand_if_available = "includes",
                    ),
                ],
            ),
        ],
    )

    if (ctx.attr.cpu == "ios_arm64" or
        ctx.attr.cpu == "ios_armv7" or
        ctx.attr.cpu == "tvos_arm64" or
        ctx.attr.cpu == "watchos_armv7k"):
        bitcode_embedded_markers_feature = feature(
            name = "bitcode_embedded_markers",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                        "objc-executable",
                        "objc++-executable",
                    ],
                    flag_groups = [flag_group(flags = ["-fembed-bitcode-marker"])],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64" or
          ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64" or
          ctx.attr.cpu == "tvos_x86_64" or
          ctx.attr.cpu == "watchos_i386" or
          ctx.attr.cpu == "x64_windows"):
        bitcode_embedded_markers_feature = feature(name = "bitcode_embedded_markers")
    else:
        bitcode_embedded_markers_feature = None

    compiler_output_flags_feature = feature(
        name = "compiler_output_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-S"],
                        expand_if_available = "output_assembly_file",
                    ),
                    flag_group(
                        flags = ["-E"],
                        expand_if_available = "output_preprocess_file",
                    ),
                    flag_group(
                        flags = ["-o", "%{output_file}"],
                        expand_if_available = "output_file",
                    ),
                ],
            ),
        ],
    )

    random_seed_feature = feature(
        name = "random_seed",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-frandom-seed=%{output_file}"],
                        expand_if_available = "output_file",
                    ),
                ],
            ),
        ],
    )

    if (ctx.attr.cpu == "ios_i386" or
        ctx.attr.cpu == "ios_x86_64" or
        ctx.attr.cpu == "tvos_x86_64" or
        ctx.attr.cpu == "watchos_i386"):
        apply_simulator_compiler_flags_feature = feature(
            name = "apply_simulator_compiler_flags",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.objc_compile, ACTION_NAMES.objcpp_compile],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-fexceptions",
                                "-fasm-blocks",
                                "-fobjc-abi-version=2",
                                "-fobjc-legacy-dispatch",
                            ],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64" or
          ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7" or
          ctx.attr.cpu == "tvos_arm64" or
          ctx.attr.cpu == "watchos_armv7k" or
          ctx.attr.cpu == "x64_windows"):
        apply_simulator_compiler_flags_feature = feature(name = "apply_simulator_compiler_flags")
    else:
        apply_simulator_compiler_flags_feature = None

    compile_all_modules_feature = feature(name = "compile_all_modules")

    dead_strip_feature = feature(
        name = "dead_strip",
        flag_sets = [
            flag_set(
                actions = _ALL_LINK_ACTIONS,
                flag_groups = [
                    flag_group(
                        flags = ["-dead_strip", "-no_dead_strip_inits_and_terms"],
                    ),
                ],
                with_features = [with_feature_set(features = ["is_not_test_target"])],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [flag_group(flags = ["-g"])],
            ),
        ],
        requires = [feature_set(features = ["opt"])],
    )

    fdo_instrument_feature = feature(
        name = "fdo_instrument",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-fprofile-generate=%{fdo_instrument_path}",
                            "-fno-data-sections",
                        ],
                        expand_if_available = "fdo_instrument_path",
                    ),
                ],
            ),
        ],
        provides = ["profile"],
    )

    lipo_feature = feature(
        name = "lipo",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-fripa"])],
            ),
        ],
        requires = [
            feature_set(features = ["autofdo"]),
            feature_set(features = ["fdo_optimize"]),
            feature_set(features = ["fdo_instrument"]),
        ],
    )

    dbg_only_flag_feature = feature(
        name = "dbg_only_flag",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.objc_compile],
                flag_groups = [flag_group(flags = ["--DBG_ONLY_FLAG"])],
            ),
        ],
    )

    fastbuild_only_flag_feature = feature(
        name = "fastbuild_only_flag",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.objc_compile],
                flag_groups = [flag_group(flags = ["--FASTBUILD_ONLY_FLAG"])],
            ),
        ],
    )

    dbg_feature = feature(name = "dbg", implies = ["dbg_only_flag"])

    is_not_test_target_feature = feature(name = "is_not_test_target")

    xcode_5_8_feature = feature(
        name = "xcode_5.8",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    "c++-header-preprocessing",
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-DXCODE_FEATURE_FOR_TESTING=xcode_5.8"],
                    ),
                ],
            ),
        ],
    )

    framework_paths_feature = feature(
        name = "framework_paths",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                    "objc-executable",
                    "objc++-executable",
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-F%{framework_paths}"],
                        iterate_over = "framework_paths",
                    ),
                ],
            ),
        ],
    )

    if (ctx.attr.cpu == "ios_arm64" or
        ctx.attr.cpu == "ios_armv7" or
        ctx.attr.cpu == "ios_i386" or
        ctx.attr.cpu == "ios_x86_64" or
        ctx.attr.cpu == "watchos_armv7k" or
        ctx.attr.cpu == "watchos_i386" or
        ctx.attr.cpu == "x64_windows"):
        apply_default_compiler_flags_feature = feature(
            name = "apply_default_compiler_flags",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.objc_compile, ACTION_NAMES.objcpp_compile],
                    flag_groups = [flag_group(flags = ["-DOS_IOS", "-fno-autolink"])],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        apply_default_compiler_flags_feature = feature(
            name = "apply_default_compiler_flags",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.objc_compile, ACTION_NAMES.objcpp_compile],
                    flag_groups = [flag_group(flags = ["-DOS_MACOSX", "-fno-autolink"])],
                ),
            ],
        )
    elif (ctx.attr.cpu == "tvos_arm64" or
          ctx.attr.cpu == "tvos_x86_64"):
        apply_default_compiler_flags_feature = feature(
            name = "apply_default_compiler_flags",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.objc_compile, ACTION_NAMES.objcpp_compile],
                    flag_groups = [flag_group(flags = ["-DOS_TVOS", "-fno-autolink"])],
                ),
            ],
        )
    else:
        apply_default_compiler_flags_feature = None

    per_object_debug_info_feature = feature(
        name = "per_object_debug_info",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-gsplit-dwarf"],
                        expand_if_available = "per_object_debug_info_file",
                    ),
                ],
            ),
        ],
    )

    symbol_counts_feature = feature(
        name = "symbol_counts",
        flag_sets = [
            flag_set(
                actions = _NON_OBJC_LINK_ACTIONS,
                flag_groups = [
                    flag_group(
                        flags = ["-Wl,--print-symbol-counts=%{symbol_counts_output}"],
                        expand_if_available = "symbol_counts_output",
                    ),
                ],
            ),
        ],
    )

    gcc_coverage_map_format_feature = feature(
        name = "gcc_coverage_map_format",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                    "objc-executable",
                    "objc++-executable",
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-fprofile-arcs", "-ftest-coverage", "-g"],
                    ),
                ],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [flag_group(flags = ["-lgcov"])],
            ),
        ],
        requires = [feature_set(features = ["coverage"])],
        provides = ["profile"],
    )

    opt_feature = feature(
        name = "opt",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [flag_group(flags = ["-g0"])],
                with_features = [
                    with_feature_set(features = ["no_generate_debug_symbols"]),
                ],
            ),
        ],
        implies = ["opt_only_flag"],
    )

    exclude_private_headers_in_module_maps_feature = feature(name = "exclude_private_headers_in_module_maps")

    no_generate_debug_symbols_feature = feature(name = "no_generate_debug_symbols")

    apply_default_warnings_feature = feature(
        name = "apply_default_warnings",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.objc_compile, ACTION_NAMES.objcpp_compile],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-Wshorten-64-to-32",
                            "-Wbool-conversion",
                            "-Wconstant-conversion",
                            "-Wduplicate-method-match",
                            "-Wempty-body",
                            "-Wenum-conversion",
                            "-Wint-conversion",
                            "-Wunreachable-code",
                            "-Wmismatched-return-types",
                            "-Wundeclared-selector",
                            "-Wuninitialized",
                            "-Wunused-function",
                            "-Wunused-variable",
                        ],
                    ),
                ],
            ),
        ],
    )

    preprocessor_defines_feature = feature(
        name = "preprocessor_defines",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-D%{preprocessor_defines}"],
                        iterate_over = "preprocessor_defines",
                    ),
                ],
            ),
        ],
    )

    if (ctx.attr.cpu == "watchos_armv7k" or
        ctx.attr.cpu == "watchos_i386"):
        include_system_dirs_feature = feature(
            name = "include_system_dirs",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                        "objc-executable",
                        "objc++-executable",
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-isysroot",
                                "%{sdk_dir}",
                                "-F%{sdk_framework_dir}",
                                "-F%{platform_developer_framework_dir}",
                            ],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64" or
          ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7" or
          ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64" or
          ctx.attr.cpu == "tvos_arm64" or
          ctx.attr.cpu == "tvos_x86_64" or
          ctx.attr.cpu == "x64_windows"):
        include_system_dirs_feature = feature(
            name = "include_system_dirs",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_module_compile,
                        ACTION_NAMES.cpp_header_parsing,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                        "objc-executable",
                        "objc++-executable",
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                    ],
                    flag_groups = [flag_group(flags = ["-isysroot", "%{sdk_dir}"])],
                ),
            ],
        )
    else:
        include_system_dirs_feature = None

    features = [
        default_compile_flags_feature,
        default_link_flags_feature,
        no_legacy_features_feature,
        fastbuild_feature,
        opt_feature,
        dbg_feature,
        compile_all_modules_feature,
        exclude_private_headers_in_module_maps_feature,
        has_configured_linker_path_feature,
        only_doth_headers_in_module_maps_feature,
        is_not_test_target_feature,
        generate_dsym_file_feature,
        no_generate_debug_symbols_feature,
        generate_linkmap_feature,
        objc_actions_feature,
        strip_debug_symbols_feature,
        symbol_counts_feature,
        shared_flag_feature,
        linkstamps_feature,
        output_execpath_flags_feature,
        runtime_root_flags_feature,
        archiver_flags_feature,
        input_param_flags_feature,
        force_pic_flags_feature,
        pch_feature,
        module_maps_feature,
        use_objc_modules_feature,
        no_enable_modules_feature,
        apply_default_warnings_feature,
        preprocessor_defines_feature,
        xcode_5_0_feature,
        xcode_5_8_feature,
        xcode_7_3_feature,
        framework_paths_feature,
        apply_default_compiler_flags_feature,
        include_system_dirs_feature,
        objc_arc_feature,
        no_objc_arc_feature,
        apple_env_feature,
        user_link_flags_feature,
        version_min_feature,
        dead_strip_feature,
        dependency_file_feature,
        random_seed_feature,
        pic_feature,
        per_object_debug_info_feature,
        includes_feature,
        include_paths_feature,
        fdo_instrument_feature,
        fdo_optimize_feature,
        autofdo_feature,
        lipo_feature,
        asan_feature,
        coverage_feature,
        llvm_coverage_map_format_feature,
        gcc_coverage_map_format_feature,
        cpp_linker_flags_feature,
        apply_implicit_frameworks_feature,
        link_cocoa_feature,
        apply_simulator_compiler_flags_feature,
        unfiltered_cxx_flags_feature,
        bitcode_embedded_markers_feature,
        bitcode_embedded_feature,
        user_compile_flags_feature,
        sysroot_feature,
        unfiltered_compile_flags_feature,
        linker_param_file_feature,
        compiler_input_flags_feature,
        compiler_output_flags_feature,
        dbg_only_flag_feature,
        fastbuild_only_flag_feature,
        opt_only_flag_feature,
    ]

    features.extend([_feature_name_to_feature[name] for name in ctx.attr.feature_names])

    artifact_name_patterns = []

    for category, values in ctx.attr.artifact_name_patterns.items():
        artifact_name_patterns.append(_get_artifact_name_pattern(category, values[0], values[1]))

    make_variables = [
        make_variable(
            name = "STACK_FRAME_UNLIMITED",
            value = "-Wframe-larger-than=100000000 -Wno-vla",
        ),
    ]

    make_variables.extend([
        make_variable(name = name, value = value)
        for name, value in ctx.attr.make_variables.items()
    ])

    cxx_builtin_include_directories = ctx.attr.cxx_builtin_include_directories

    if (ctx.attr.cpu == "x64_windows"):
        tool_paths = [
            tool_path(name = "ar", path = "<tool_dir>/ar_wrapper"),
            tool_path(name = "compat-ld", path = "<tool_dir>/ld"),
            tool_path(name = "cpp", path = "<tool_dir>/cpp"),
            tool_path(name = "gcov", path = "<tool_dir>/gcov"),
            tool_path(name = "gcc", path = "<tool_dir>/clang"),
            tool_path(name = "ld", path = "<tool_dir>/ld"),
            tool_path(name = "nm", path = "<tool_dir>/nm"),
            tool_path(name = "strip", path = "<tool_dir>/strip"),
            tool_path(name = "dwp", path = "/usr/bin/dwp"),
            tool_path(name = "objdump", path = "/usr/bin/objdump"),
        ]
    elif (ctx.attr.cpu == "ios_arm64" or
          ctx.attr.cpu == "ios_armv7"):
        tool_paths = [
            tool_path(name = "ar", path = "ios/ar_wrapper"),
            tool_path(name = "compat-ld", path = "ios/ld"),
            tool_path(name = "cpp", path = "ios/cpp"),
            tool_path(name = "gcov", path = "ios/gcov"),
            tool_path(name = "gcc", path = "ios/clang"),
            tool_path(name = "ld", path = "ios/ld"),
            tool_path(name = "nm", path = "ios/nm"),
            tool_path(name = "strip", path = "ios/strip"),
            tool_path(name = "dwp", path = "/usr/bin/dwp"),
            tool_path(name = "objdump", path = "/usr/bin/objdump"),
        ]
    elif (ctx.attr.cpu == "ios_i386" or
          ctx.attr.cpu == "ios_x86_64"):
        tool_paths = [
            tool_path(name = "ar", path = "iossim/ar_wrapper"),
            tool_path(name = "compat-ld", path = "iossim/ld"),
            tool_path(name = "cpp", path = "iossim/cpp"),
            tool_path(name = "gcov", path = "iossim/gcov"),
            tool_path(name = "gcc", path = "iossim/clang"),
            tool_path(name = "ld", path = "iossim/ld"),
            tool_path(name = "nm", path = "iossim/nm"),
            tool_path(name = "strip", path = "iossim/strip"),
            tool_path(name = "dwp", path = "/usr/bin/dwp"),
            tool_path(name = "objdump", path = "/usr/bin/objdump"),
        ]
    elif (ctx.attr.cpu == "darwin_x86_64"):
        tool_paths = [
            tool_path(name = "ar", path = "mac/ar_wrapper"),
            tool_path(name = "compat-ld", path = "mac/ld"),
            tool_path(name = "cpp", path = "mac/cpp"),
            tool_path(name = "gcov", path = "mac/gcov"),
            tool_path(name = "gcc", path = "mac/clang"),
            tool_path(name = "ld", path = "mac/ld"),
            tool_path(name = "nm", path = "mac/nm"),
            tool_path(name = "strip", path = "mac/strip"),
            tool_path(name = "dwp", path = "/usr/bin/dwp"),
            tool_path(name = "objdump", path = "/usr/bin/objdump"),
        ]
    elif (ctx.attr.cpu == "tvos_arm64"):
        tool_paths = [
            tool_path(name = "ar", path = "tvos/ar_wrapper"),
            tool_path(name = "compat-ld", path = "tvos/ld"),
            tool_path(name = "cpp", path = "tvos/cpp"),
            tool_path(name = "gcov", path = "tvos/gcov"),
            tool_path(name = "gcc", path = "tvos/clang"),
            tool_path(name = "ld", path = "tvos/ld"),
            tool_path(name = "nm", path = "tvos/nm"),
            tool_path(name = "strip", path = "tvos/strip"),
            tool_path(name = "dwp", path = "/usr/bin/dwp"),
            tool_path(name = "objdump", path = "/usr/bin/objdump"),
        ]
    elif (ctx.attr.cpu == "tvos_x86_64"):
        tool_paths = [
            tool_path(name = "ar", path = "tvsim/ar_wrapper"),
            tool_path(name = "compat-ld", path = "tvsim/ld"),
            tool_path(name = "cpp", path = "tvsim/cpp"),
            tool_path(name = "gcov", path = "tvsim/gcov"),
            tool_path(name = "gcc", path = "tvsim/clang"),
            tool_path(name = "ld", path = "tvsim/ld"),
            tool_path(name = "nm", path = "tvsim/nm"),
            tool_path(name = "strip", path = "tvsim/strip"),
            tool_path(name = "dwp", path = "/usr/bin/dwp"),
            tool_path(name = "objdump", path = "/usr/bin/objdump"),
        ]
    elif (ctx.attr.cpu == "watchos_armv7k"):
        tool_paths = [
            tool_path(name = "ar", path = "watchos/ar_wrapper"),
            tool_path(name = "compat-ld", path = "watchos/ld"),
            tool_path(name = "cpp", path = "watchos/cpp"),
            tool_path(name = "gcov", path = "watchos/gcov"),
            tool_path(name = "gcc", path = "watchos/clang"),
            tool_path(name = "ld", path = "watchos/ld"),
            tool_path(name = "nm", path = "watchos/nm"),
            tool_path(name = "strip", path = "watchos/strip"),
            tool_path(name = "dwp", path = "/usr/bin/dwp"),
            tool_path(name = "objdump", path = "/usr/bin/objdump"),
        ]
    elif (ctx.attr.cpu == "watchos_i386"):
        tool_paths = [
            tool_path(name = "ar", path = "watchsim/ar_wrapper"),
            tool_path(name = "compat-ld", path = "watchsim/ld"),
            tool_path(name = "cpp", path = "watchsim/cpp"),
            tool_path(name = "gcov", path = "watchsim/gcov"),
            tool_path(name = "gcc", path = "watchsim/clang"),
            tool_path(name = "ld", path = "watchsim/ld"),
            tool_path(name = "nm", path = "watchsim/nm"),
            tool_path(name = "strip", path = "watchsim/strip"),
            tool_path(name = "dwp", path = "/usr/bin/dwp"),
            tool_path(name = "objdump", path = "/usr/bin/objdump"),
        ]
    else:
        tool_paths = []

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
        "cpu": attr.string(mandatory = True, values = [
            "darwin_x86_64",
            "ios_arm64",
            "ios_armv7",
            "ios_i386",
            "ios_x86_64",
            "tvos_arm64",
            "tvos_x86_64",
            "watchos_armv7k",
            "watchos_i386",
            "x64_windows",
        ]),
        "compiler": attr.string(mandatory = True),
        "toolchain_identifier": attr.string(mandatory = True),
        "host_system_name": attr.string(mandatory = True),
        "target_system_name": attr.string(mandatory = True),
        "target_libc": attr.string(mandatory = True),
        "abi_version": attr.string(mandatory = True),
        "abi_libc_version": attr.string(mandatory = True),
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
