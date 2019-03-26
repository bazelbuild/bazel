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

load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
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

_VALID_CPUS = [
    "armeabi-v7a",
    "darwin_x86_64",
    "ios_arm64",
    "ios_arm64e",
    "ios_armv7",
    "ios_i386",
    "ios_x86_64",
    "tvos_arm64",
    "tvos_x86_64",
    "watchos_arm64_32",
    "watchos_armv7k",
    "watchos_i386",
    "watchos_x86_64",
]

def _version_min_for_cpu(cpu):
    if cpu == "ios_i386" or cpu == "ios_x86_64":
        return "-mios-simulator-version-min"
    elif cpu == "ios_arm64" or cpu == "ios_arm64e" or cpu == "ios_armv7":
        return "-miphoneos-version-min"
    elif cpu == "tvos_x86_64":
        return "-mtvos-simulator-version-min"
    elif cpu == "tvos_arm64":
        return "-mtvos-version-min"
    elif cpu == "watchos_i386" or cpu == "watchos_x86_64":
        return "-mwatchos-simulator-version-min"
    elif cpu == "watchos_armv7k" or cpu == "watchos_arm64_32":
        return "-mwatchos-version-min"
    elif (cpu == "darwin_x86_64"):
        return "-mmacosx-version-min"
    else:
        fail("Unhandled version min cpu {}".format(cpu))

def _arch_for_cpu(cpu):
    if cpu == "armeabi-v7a" or cpu == "watchos_arm64_32":
        return "<architecture>"
    elif cpu == "ios_arm64" or cpu == "tvos_arm64":
        return "arm64"
    elif cpu == "ios_arm64e":
        return "arm64e"
    elif cpu == "ios_armv7":
        return "armv7"
    elif cpu == "watchos_armv7k":
        return "armv7k"
    elif cpu == "ios_i386" or cpu == "watchos_i386":
        return "i386"
    elif cpu == "darwin_x86_64" or cpu == "ios_x86_64" or cpu == "tvos_x86_64" or cpu == "watchos_x86_64":
        return "x86_64"
    else:
        fail("Unhandled architecture cpu {}".format(cpu))

def _impl(ctx):
    if ctx.attr.cpu not in _VALID_CPUS:
        fail("Unhandled CPU {}".format(ctx.attr.cpu))

    if ctx.attr.cpu == "armeabi-v7a":
        toolchain_identifier = "stub_armeabi-v7a"
    else:
        toolchain_identifier = ctx.attr.cpu

    if ctx.attr.cpu == "armeabi-v7a":
        host_system_name = "armeabi-v7a"
    else:
        host_system_name = "x86_64-apple-macosx"

    if (ctx.attr.cpu == "ios_arm64"):
        target_system_name = "arm64-apple-ios"
    elif (ctx.attr.cpu == "tvos_arm64"):
        target_system_name = "arm64-apple-tvos"
    elif (ctx.attr.cpu == "watchos_arm64_32"):
        target_system_name = "arm64_32-apple-watchos"
    elif (ctx.attr.cpu == "ios_arm64e"):
        target_system_name = "arm64e-apple-ios"
    elif (ctx.attr.cpu == "armeabi-v7a"):
        target_system_name = "armeabi-v7a"
    elif (ctx.attr.cpu == "ios_armv7"):
        target_system_name = "armv7-apple-ios"
    elif (ctx.attr.cpu == "watchos_armv7k"):
        target_system_name = "armv7-apple-watchos"
    elif (ctx.attr.cpu == "ios_i386"):
        target_system_name = "i386-apple-ios"
    elif (ctx.attr.cpu == "watchos_i386"):
        target_system_name = "i386-apple-watchos"
    elif (ctx.attr.cpu == "ios_x86_64"):
        target_system_name = "x86_64-apple-ios"
    elif (ctx.attr.cpu == "darwin_x86_64"):
        target_system_name = "x86_64-apple-macosx"
    elif (ctx.attr.cpu == "tvos_x86_64"):
        target_system_name = "x86_64-apple-tvos"
    elif (ctx.attr.cpu == "watchos_x86_64"):
        target_system_name = "x86_64-apple-watchos"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "armeabi-v7a"):
        target_libc = "armeabi-v7a"
    elif (ctx.attr.cpu == "ios_arm64"
        or ctx.attr.cpu == "ios_arm64e"
        or ctx.attr.cpu == "ios_armv7"
        or ctx.attr.cpu == "ios_i386"
        or ctx.attr.cpu == "ios_x86_64"):
        target_libc = "ios"
    elif (ctx.attr.cpu == "darwin_x86_64"):
        target_libc = "macosx"
    elif (ctx.attr.cpu == "tvos_arm64"
        or ctx.attr.cpu == "tvos_x86_64"):
        target_libc = "tvos"
    elif (ctx.attr.cpu == "watchos_arm64_32"
        or ctx.attr.cpu == "watchos_armv7k"
        or ctx.attr.cpu == "watchos_i386"
        or ctx.attr.cpu == "watchos_x86_64"):
        target_libc = "watchos"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "armeabi-v7a"):
        abi_version = "armeabi-v7a"
    elif (ctx.attr.cpu == "darwin_x86_64"):
        abi_version = "darwin_x86_64"
    else:
        abi_version = "local"

    if (ctx.attr.cpu == "armeabi-v7a"):
        abi_libc_version = "armeabi-v7a"
    elif (ctx.attr.cpu == "darwin_x86_64"):
        abi_libc_version = "darwin_x86_64"
    else:
        abi_libc_version = "local"

    all_compile_actions = [
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.assemble,
        ACTION_NAMES.preprocess_assemble,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.cpp_module_codegen,
        ACTION_NAMES.clif_match,
        ACTION_NAMES.lto_backend,
    ]

    all_cpp_compile_actions = [
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.cpp_module_codegen,
        ACTION_NAMES.clif_match,
    ]

    preprocessor_compile_actions = [
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.preprocess_assemble,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.clif_match,
    ]

    codegen_compile_actions = [
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.assemble,
        ACTION_NAMES.preprocess_assemble,
        ACTION_NAMES.cpp_module_codegen,
        ACTION_NAMES.lto_backend,
    ]

    all_link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

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
        tools = [tool(path = "/usr/bin/strip")],
    )

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
                path = "wrapped_clang",
                execution_requirements = ["requires-darwin"],
            ),
        ],
    )

    objc_compile_action = action_config(
        action_name = ACTION_NAMES.objc_compile,
        flag_sets = [
            flag_set(
                flag_groups = [flag_group(flags = ["-arch", _arch_for_cpu(ctx.attr.cpu)])],
            ),
        ],
        implies = [
            "compiler_input_flags",
            "compiler_output_flags",
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
            "apply_simulator_compiler_flags",
        ],
        tools = [
            tool(
                path = "wrapped_clang",
                execution_requirements = ["requires-darwin"],
            ),
        ],
    )

    objcpp_executable_action = action_config(
        action_name = "objc++-executable",
        flag_sets = [
            flag_set(
                flag_groups = [
                    flag_group(flags = ["-stdlib=libc++", "-std=gnu++11"]),
                    flag_group(flags = ["-arch", _arch_for_cpu(ctx.attr.cpu)]),
                    flag_group(
                        flags = [
                            "-Xlinker",
                            "-objc_abi_version",
                            "-Xlinker",
                            "2",
                            "-fobjc-link-runtime",
                            "-ObjC",
                        ],
                    ),
                    flag_group(
                        flags = ["-framework", "%{framework_names}"],
                        iterate_over = "framework_names",
                    ),
                    flag_group(
                        flags = ["-weak_framework", "%{weak_framework_names}"],
                        iterate_over = "weak_framework_names",
                    ),
                    flag_group(
                        flags = ["-l%{library_names}"],
                        iterate_over = "library_names",
                    ),
                    flag_group(flags = ["-filelist", "%{filelist}"]),
                    flag_group(flags = ["-o", "%{linked_binary}"]),
                    flag_group(
                        flags = ["-force_load", "%{force_load_exec_paths}"],
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
            "strip_debug_symbols",
            "apple_env",
            "apply_implicit_frameworks",
        ],
        tools = [
            tool(
                path = "wrapped_clang_pp",
                execution_requirements = ["requires-darwin"],
            ),
        ],
    )

    if (ctx.attr.cpu == "tvos_arm64"
        or ctx.attr.cpu == "tvos_x86_64"):
        cpp_link_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_dynamic_library,
            implies = [
                "contains_objc_source",
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
                "sysroot",
                "cpp_linker_flags",
            ],
            tools = [
                tool(
                    path = "cc_wrapper.sh",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "armeabi-v7a"
        or ctx.attr.cpu == "darwin_x86_64"
        or ctx.attr.cpu == "ios_arm64"
        or ctx.attr.cpu == "ios_arm64e"
        or ctx.attr.cpu == "ios_armv7"
        or ctx.attr.cpu == "ios_i386"
        or ctx.attr.cpu == "ios_x86_64"
        or ctx.attr.cpu == "watchos_arm64_32"
        or ctx.attr.cpu == "watchos_armv7k"
        or ctx.attr.cpu == "watchos_i386"
        or ctx.attr.cpu == "watchos_x86_64"):
        cpp_link_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_dynamic_library,
            implies = [
                "contains_objc_source",
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
                "sysroot",
            ],
            tools = [
                tool(
                    path = "cc_wrapper.sh",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )

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
                path = "wrapped_ar",
                execution_requirements = ["requires-darwin"],
            ),
        ],
    )

    c_compile_action = action_config(
        action_name = ACTION_NAMES.c_compile,
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
                path = "wrapped_clang",
                execution_requirements = ["requires-darwin"],
            ),
        ],
    )

    cpp_compile_action = action_config(
        action_name = ACTION_NAMES.cpp_compile,
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
                path = "wrapped_clang",
                execution_requirements = ["requires-darwin"],
            ),
        ],
    )

    objcpp_compile_action = action_config(
        action_name = ACTION_NAMES.objcpp_compile,
        flag_sets = [
            flag_set(
                flag_groups = [
                    flag_group(
                        flags = ["-arch", _arch_for_cpu(ctx.attr.cpu), "-stdlib=libc++", "-std=gnu++11"],
                    ),
                ],
            ),
        ],
        implies = [
            "compiler_input_flags",
            "compiler_output_flags",
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
            "apply_simulator_compiler_flags",
        ],
        tools = [
            tool(
                path = "wrapped_clang",
                execution_requirements = ["requires-darwin"],
            ),
        ],
    )

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
                path = "wrapped_clang",
                execution_requirements = ["requires-darwin"],
            ),
        ],
    )

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
                path = "wrapped_clang",
                execution_requirements = ["requires-darwin"],
            ),
        ],
    )

    objc_archive_action = action_config(
        action_name = "objc-archive",
        flag_sets = [
            flag_set(
                flag_groups = [
                    flag_group(
                        flags = [
                            "-no_warning_for_no_symbols",
                            "-static",
                            "-filelist",
                            "%{obj_list_path}",
                            "-arch_only",
                            _arch_for_cpu(ctx.attr.cpu),
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
                path = "libtool",
                execution_requirements = ["requires-darwin"],
            ),
        ],
    )

    objc_executable_action = action_config(
        action_name = "objc-executable",
        flag_sets = [
            flag_set(
                flag_groups = [
                    flag_group(
                        flags = [
                            "-Xlinker",
                            "-objc_abi_version",
                            "-Xlinker",
                            "2",
                            "-fobjc-link-runtime",
                            "-ObjC",
                        ],
                    ),
                ],
                with_features = [with_feature_set(not_features = ["kernel_extension"])],
            ),
            flag_set(
                flag_groups = [
                    flag_group(flags = ["-arch", _arch_for_cpu(ctx.attr.cpu)]),
                    flag_group(
                        flags = ["-framework", "%{framework_names}"],
                        iterate_over = "framework_names",
                    ),
                    flag_group(
                        flags = ["-weak_framework", "%{weak_framework_names}"],
                        iterate_over = "weak_framework_names",
                    ),
                    flag_group(
                        flags = ["-l%{library_names}"],
                        iterate_over = "library_names",
                    ),
                    flag_group(flags = ["-filelist", "%{filelist}"]),
                    flag_group(flags = ["-o", "%{linked_binary}"]),
                    flag_group(
                        flags = ["-force_load", "%{force_load_exec_paths}"],
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
            "strip_debug_symbols",
            "apple_env",
            "apply_implicit_frameworks",
        ],
        tools = [
            tool(
                path = "wrapped_clang",
                execution_requirements = ["requires-darwin"],
            ),
        ],
    )

    if (ctx.attr.cpu == "tvos_arm64"
        or ctx.attr.cpu == "tvos_x86_64"):
        cpp_link_executable_action = action_config(
            action_name = ACTION_NAMES.cpp_link_executable,
            implies = [
                "contains_objc_source",
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
                "sysroot",
                "cpp_linker_flags",
            ],
            tools = [
                tool(
                    path = "cc_wrapper.sh",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "armeabi-v7a"
        or ctx.attr.cpu == "darwin_x86_64"
        or ctx.attr.cpu == "ios_arm64"
        or ctx.attr.cpu == "ios_arm64e"
        or ctx.attr.cpu == "ios_armv7"
        or ctx.attr.cpu == "ios_i386"
        or ctx.attr.cpu == "ios_x86_64"
        or ctx.attr.cpu == "watchos_arm64_32"
        or ctx.attr.cpu == "watchos_armv7k"
        or ctx.attr.cpu == "watchos_i386"
        or ctx.attr.cpu == "watchos_x86_64"):
        cpp_link_executable_action = action_config(
            action_name = ACTION_NAMES.cpp_link_executable,
            implies = [
                "contains_objc_source",
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
                "sysroot",
            ],
            tools = [
                tool(
                    path = "cc_wrapper.sh",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )

    linkstamp_compile_action = action_config(
        action_name = ACTION_NAMES.linkstamp_compile,
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
                path = "wrapped_clang",
                execution_requirements = ["requires-darwin"],
            ),
        ],
    )

    if (ctx.attr.cpu == "tvos_arm64"
        or ctx.attr.cpu == "tvos_x86_64"):
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
                    path = "wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "armeabi-v7a"
        or ctx.attr.cpu == "darwin_x86_64"
        or ctx.attr.cpu == "ios_arm64"
        or ctx.attr.cpu == "ios_arm64e"
        or ctx.attr.cpu == "ios_armv7"
        or ctx.attr.cpu == "ios_i386"
        or ctx.attr.cpu == "ios_x86_64"
        or ctx.attr.cpu == "watchos_arm64_32"
        or ctx.attr.cpu == "watchos_armv7k"
        or ctx.attr.cpu == "watchos_i386"
        or ctx.attr.cpu == "watchos_x86_64"):
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
                    path = "wrapped_clang",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )

    if (ctx.attr.cpu == "tvos_arm64"
        or ctx.attr.cpu == "tvos_x86_64"):
        cpp_link_nodeps_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
            implies = [
                "contains_objc_source",
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
                "sysroot",
                "cpp_linker_flags",
            ],
            tools = [
                tool(
                    path = "cc_wrapper.sh",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )
    elif (ctx.attr.cpu == "armeabi-v7a"
        or ctx.attr.cpu == "darwin_x86_64"
        or ctx.attr.cpu == "ios_arm64"
        or ctx.attr.cpu == "ios_arm64e"
        or ctx.attr.cpu == "ios_armv7"
        or ctx.attr.cpu == "ios_i386"
        or ctx.attr.cpu == "ios_x86_64"
        or ctx.attr.cpu == "watchos_arm64_32"
        or ctx.attr.cpu == "watchos_armv7k"
        or ctx.attr.cpu == "watchos_i386"
        or ctx.attr.cpu == "watchos_x86_64"):
        cpp_link_nodeps_dynamic_library_action = action_config(
            action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
            implies = [
                "contains_objc_source",
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
                "sysroot",
            ],
            tools = [
                tool(
                    path = "cc_wrapper.sh",
                    execution_requirements = ["requires-darwin"],
                ),
            ],
        )

    objc_fully_link_action = action_config(
        action_name = "objc-fully-link",
        flag_sets = [
            flag_set(
                flag_groups = [
                    flag_group(
                        flags = [
                            "-no_warning_for_no_symbols",
                            "-static",
                            "-arch_only",
                            _arch_for_cpu(ctx.attr.cpu),
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
                path = "libtool",
                execution_requirements = ["requires-darwin"],
            ),
        ],
    )

    if (ctx.attr.cpu == "armeabi-v7a"):
        objcopy_embed_data_action = action_config(
            action_name = "objcopy_embed_data",
            enabled = True,
            tools = [tool(path = "/bin/false")],
        )
    else:
        objcopy_embed_data_action = action_config(
            action_name = "objcopy_embed_data",
            enabled = True,
            tools = [tool(path = "/usr/bin/objcopy")],
        )

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
        objcopy_embed_data_action,
    ]

    if (ctx.attr.cpu == "armeabi-v7a"
        or ctx.attr.cpu == "ios_arm64"
        or ctx.attr.cpu == "ios_arm64e"
        or ctx.attr.cpu == "ios_armv7"
        or ctx.attr.cpu == "ios_i386"
        or ctx.attr.cpu == "ios_x86_64"
        or ctx.attr.cpu == "watchos_arm64_32"
        or ctx.attr.cpu == "watchos_armv7k"
        or ctx.attr.cpu == "watchos_i386"
        or ctx.attr.cpu == "watchos_x86_64"):
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
    elif (ctx.attr.cpu == "tvos_arm64"
        or ctx.attr.cpu == "tvos_x86_64"):
        apply_default_compiler_flags_feature = feature(
            name = "apply_default_compiler_flags",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.objc_compile, ACTION_NAMES.objcpp_compile],
                    flag_groups = [flag_group(flags = ["-DOS_TVOS", "-fno-autolink"])],
                ),
            ],
        )

    dynamic_linking_mode_feature = feature(name = "dynamic_linking_mode")

    compile_all_modules_feature = feature(name = "compile_all_modules")

    runtime_root_flags_feature = feature(
        name = "runtime_root_flags",
        flag_sets = [
            flag_set(
                actions = all_link_actions +
                    [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}",
                        ],
                        iterate_over = "runtime_library_search_directories",
                        expand_if_available = "runtime_library_search_directories",
                    ),
                ],
            ),
            flag_set(
                actions = all_link_actions +
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
                actions = all_link_actions +
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

    if (ctx.attr.cpu == "tvos_arm64"
        or ctx.attr.cpu == "tvos_x86_64"):
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
    elif (ctx.attr.cpu == "armeabi-v7a"
        or ctx.attr.cpu == "darwin_x86_64"
        or ctx.attr.cpu == "ios_arm64"
        or ctx.attr.cpu == "ios_arm64e"
        or ctx.attr.cpu == "ios_armv7"
        or ctx.attr.cpu == "ios_i386"
        or ctx.attr.cpu == "ios_x86_64"
        or ctx.attr.cpu == "watchos_arm64_32"
        or ctx.attr.cpu == "watchos_armv7k"
        or ctx.attr.cpu == "watchos_i386"
        or ctx.attr.cpu == "watchos_x86_64"):
        unfiltered_cxx_flags_feature = feature(name = "unfiltered_cxx_flags")

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

    strip_debug_symbols_feature = feature(
        name = "strip_debug_symbols",
        flag_sets = [
            flag_set(
                actions = all_link_actions +
                    ["objc-executable", "objc++-executable"],
                flag_groups = [
                    flag_group(
                        flags = ["-Wl,-S"],
                        expand_if_available = "strip_debug_symbols",
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

    if (ctx.attr.cpu == "ios_i386"
        or ctx.attr.cpu == "ios_x86_64"
        or ctx.attr.cpu == "tvos_x86_64"
        or ctx.attr.cpu == "watchos_i386"
        or ctx.attr.cpu == "watchos_x86_64"):
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
    elif (ctx.attr.cpu == "armeabi-v7a"
        or ctx.attr.cpu == "darwin_x86_64"
        or ctx.attr.cpu == "ios_arm64"
        or ctx.attr.cpu == "ios_arm64e"
        or ctx.attr.cpu == "ios_armv7"
        or ctx.attr.cpu == "tvos_arm64"
        or ctx.attr.cpu == "watchos_arm64_32"
        or ctx.attr.cpu == "watchos_armv7k"):
        apply_simulator_compiler_flags_feature = feature(name = "apply_simulator_compiler_flags")

    supports_pic_feature = feature(name = "supports_pic", enabled = True)

    fastbuild_feature = feature(name = "fastbuild")

    no_legacy_features_feature = feature(name = "no_legacy_features")

    symbol_counts_feature = feature(
        name = "symbol_counts",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["-Wl,--print-symbol-counts=%{symbol_counts_output}"],
                        expand_if_available = "symbol_counts_output",
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
                actions = all_link_actions +
                    ["objc-executable", "objc++-executable"],
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

    if (ctx.attr.cpu == "armeabi-v7a"
        or ctx.attr.cpu == "ios_arm64"
        or ctx.attr.cpu == "ios_arm64e"
        or ctx.attr.cpu == "ios_armv7"
        or ctx.attr.cpu == "ios_i386"
        or ctx.attr.cpu == "ios_x86_64"
        or ctx.attr.cpu == "tvos_arm64"
        or ctx.attr.cpu == "tvos_x86_64"
        or ctx.attr.cpu == "watchos_arm64_32"
        or ctx.attr.cpu == "watchos_armv7k"
        or ctx.attr.cpu == "watchos_i386"
        or ctx.attr.cpu == "watchos_x86_64"):
        contains_objc_source_feature = feature(
            name = "contains_objc_source",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.cpp_link_executable,
                    ],
                    flag_groups = [flag_group(flags = ["-fobjc-link-runtime"])],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.cpp_link_executable,
                    ],
                    flag_groups = [flag_group(flags = ["-framework", "UIKit"])],
                ),
            ],
        )
    elif (ctx.attr.cpu == "darwin_x86_64"):
        contains_objc_source_feature = feature(
            name = "contains_objc_source",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.cpp_link_executable,
                    ],
                    flag_groups = [flag_group(flags = ["-fobjc-link-runtime"])],
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
                flag_groups = [flag_group(flags = ["--coverage"])],
            ),
        ],
        requires = [feature_set(features = ["coverage"])],
    )

    if ctx.attr.cpu == "darwin_x86_64":
        default_link_flags_feature = feature(
            name = "default_link_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions +
                        ["objc-executable", "objc++-executable"],
                    flag_groups = [
                        flag_group(
                            flags = ["-no-canonical-prefixes"],
                        ),
                    ],
                ),
                flag_set(
                    actions = [
                      ACTION_NAMES.cpp_link_dynamic_library,
                      ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ],
                    flag_groups = [flag_group(flags = ["-undefined", "dynamic_lookup"])],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.cpp_link_executable,
                        "objc-executable",
                        "objc++-executable",
                    ],
                    flag_groups = [flag_group(flags = ["-undefined", "dynamic_lookup"])],
                    with_features = [with_feature_set(features = ["dynamic_linking_mode"])],
                ),
            ],
        )
    else:
        default_link_flags_feature = feature(
            name = "default_link_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions +
                        ["objc-executable", "objc++-executable"],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-no-canonical-prefixes",
                                "-target",
                                target_system_name,
                            ],
                        ),
                    ],
                ),
            ],
        )

    output_execpath_flags_feature = feature(
        name = "output_execpath_flags",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["-o", "%{output_execpath}"],
                        expand_if_available = "output_execpath",
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
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [
                    flag_group(flags = ["-fPIC"], expand_if_available = "pic"),
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

    if ctx.attr.cpu == "armeabi-v7a":
        # This stub doesn't have a sensible value for this feature
        version_min_feature = feature(name = "version_min")
    else:
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
                    flag_groups = [flag_group(flags = [_version_min_for_cpu(ctx.attr.cpu) + "=%{version_min}"])],
                ),
            ],
        )

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

    opt_feature = feature(name = "opt")

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

    coverage_feature = feature(name = "coverage")

    if (ctx.attr.cpu == "watchos_arm64_32"
        or ctx.attr.cpu == "watchos_armv7k"
        or ctx.attr.cpu == "watchos_i386"
        or ctx.attr.cpu == "watchos_x86_64"):
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
    elif (ctx.attr.cpu == "armeabi-v7a"
        or ctx.attr.cpu == "darwin_x86_64"
        or ctx.attr.cpu == "ios_arm64"
        or ctx.attr.cpu == "ios_arm64e"
        or ctx.attr.cpu == "ios_armv7"
        or ctx.attr.cpu == "ios_i386"
        or ctx.attr.cpu == "ios_x86_64"
        or ctx.attr.cpu == "tvos_arm64"
        or ctx.attr.cpu == "tvos_x86_64"):
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

    input_param_flags_feature = feature(
        name = "input_param_flags",
        flag_sets = [
            flag_set(
                actions = all_link_actions +
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
                actions = all_link_actions +
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
                actions = all_link_actions +
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
                actions = all_link_actions +
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
                actions = all_link_actions +
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
                    ACTION_NAMES.linkstamp_compile,
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

    if (ctx.attr.cpu == "armeabi-v7a"
        or ctx.attr.cpu == "ios_arm64"
        or ctx.attr.cpu == "ios_arm64e"
        or ctx.attr.cpu == "ios_armv7"
        or ctx.attr.cpu == "ios_i386"
        or ctx.attr.cpu == "ios_x86_64"
        or ctx.attr.cpu == "tvos_arm64"
        or ctx.attr.cpu == "tvos_x86_64"
        or ctx.attr.cpu == "watchos_arm64_32"
        or ctx.attr.cpu == "watchos_armv7k"
        or ctx.attr.cpu == "watchos_i386"
        or ctx.attr.cpu == "watchos_x86_64"):
        apply_implicit_frameworks_feature = feature(
            name = "apply_implicit_frameworks",
            flag_sets = [
                flag_set(
                    actions = ["objc-executable", "objc++-executable"],
                    flag_groups = [
                        flag_group(
                            flags = ["-framework", "Foundation", "-framework", "UIKit"],
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
                    flag_groups = [flag_group(flags = ["-framework", "Foundation"])],
                    with_features = [with_feature_set(not_features = ["kernel_extension"])],
                ),
            ],
        )

    dbg_feature = feature(name = "dbg")

    has_configured_linker_path_feature = feature(name = "has_configured_linker_path")

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
                    ACTION_NAMES.linkstamp_compile,
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

    link_libcpp_feature = feature(
        name = "link_libc++",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions +
                    ["objc-executable", "objc++-executable"],
                flag_groups = [flag_group(flags = ["-lc++"])],
                with_features = [with_feature_set(not_features = ["kernel_extension"])],
            ),
        ],
    )

    supports_dynamic_linker_feature = feature(name = "supports_dynamic_linker", enabled = True)

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

    module_maps_feature = feature(name = "module_maps", enabled = True)

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
                    ACTION_NAMES.linkstamp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-no-canonical-prefixes",
                            "-Wno-builtin-macro-redefined",
                            "-D__DATE__=\"redacted\"",
                            "-D__TIMESTAMP__=\"redacted\"",
                            "-D__TIME__=\"redacted\"",
                            "-target",
                            target_system_name,
                        ],
                    ),
                ],
            ),
        ],
    )

    linker_param_file_feature = feature(
        name = "linker_param_file",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
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

    archiver_flags_feature = feature(
        name = "archiver_flags",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["rcS", "%{output_execpath}"],
                        expand_if_available = "output_execpath",
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

    cpp_linker_flags_feature = feature(
        name = "cpp_linker_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-lc++", "-target", target_system_name],
                    ),
                ],
            ),
        ],
    )

    exclude_private_headers_in_module_maps_feature = feature(name = "exclude_private_headers_in_module_maps")

    debug_prefix_map_pwd_is_dot_feature = feature(
        name = "debug_prefix_map_pwd_is_dot",
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
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [flag_group(flags = ["DEBUG_PREFIX_MAP_PWD=."])],
            ),
        ],
    )

    linkstamps_feature = feature(
        name = "linkstamps",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
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
                    ACTION_NAMES.linkstamp_compile,
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

    only_doth_headers_in_module_maps_feature = feature(name = "only_doth_headers_in_module_maps")

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
                            "-D_FORTIFY_SOURCE=1",
                            "-fstack-protector",
                            "-fcolor-diagnostics",
                            "-Wall",
                            "-Wthread-safety",
                            "-Wself-assign",
                            "-fno-omit-frame-pointer",
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
                        flags = [
                            "-g0",
                            "-O2",
                            "-D_FORTIFY_SOURCE=1",
                            "-DNDEBUG",
                            "-ffunction-sections",
                            "-fdata-sections",
                            "-DNS_BLOCK_ASSERTIONS=1",
                        ],
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
                flag_groups = [flag_group(flags = ["-g"])],
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
                flag_groups = [flag_group(flags = ["-std=c++11"])],
            ),
        ],
    )

    objcopy_embed_flags_feature = feature(
        name = "objcopy_embed_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = ["objcopy_embed_data"],
                flag_groups = [flag_group(flags = ["-I", "binary"])],
            ),
        ],
    )

    dead_strip_feature = feature(
        name = "dead_strip",
        flag_sets = [
            flag_set(
                actions = all_link_actions +
                    ["objc-executable", "objc++-executable"],
                flag_groups = [
                    flag_group(
                        flags = ["-dead_strip", "-no_dead_strip_inits_and_terms"],
                    ),
                ],
            ),
        ],
        requires = [feature_set(features = ["opt"])],
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

    if (ctx.attr.cpu == "darwin_x86_64"):
        kernel_extension_feature = feature(
            name = "kernel_extension",
            flag_sets = [
                flag_set(
                    actions = ["objc-executable", "objc++-executable"],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-nostdlib",
                                "-lkmod",
                                "-lkmodc++",
                                "-lcc_kext",
                                "-Xlinker",
                                "-kext",
                            ],
                        ),
                    ],
                ),
            ],
        )
    else:
        kernel_extension_feature = feature(name = "kernel_extension")

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
                    ACTION_NAMES.linkstamp_compile,
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

    if (ctx.attr.cpu == "darwin_x86_64"):
        link_cocoa_feature = feature(
            name = "link_cocoa",
            flag_sets = [
                flag_set(
                    actions = ["objc-executable", "objc++-executable"],
                    flag_groups = [flag_group(flags = ["-framework", "Cocoa"])],
                ),
            ],
        )
    else:
        link_cocoa_feature = feature(name = "link_cocoa")

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
                    ACTION_NAMES.linkstamp_compile,
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

    headerpad_feature = feature(
        name = "headerpad",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions + [
                    ACTION_NAMES.objc_executable,
                    ACTION_NAMES.objcpp_executable,
                ],
                flag_groups = [flag_group(flags = ["-headerpad_max_install_names"])],
                with_features = [with_feature_set(not_features = [
                    "bitcode_embedded",
                    "bitcode_embedded_markers",
                ])],
            ),
        ],
    )

    if (ctx.attr.cpu == "ios_arm64"
        or ctx.attr.cpu == "ios_arm64e"
        or ctx.attr.cpu == "ios_armv7"
        or ctx.attr.cpu == "tvos_arm64"
        or ctx.attr.cpu == "watchos_arm64_32"
        or ctx.attr.cpu == "watchos_armv7k"
        or ctx.attr.cpu == "darwin_x86_64"):
        bitcode_embedded_feature = feature(
            name = "bitcode_embedded",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [flag_group(flags = ["-fembed-bitcode"])],
                ),
                flag_set(
                    actions = all_link_actions + [
                        ACTION_NAMES.objc_executable,
                        ACTION_NAMES.objcpp_executable,
                    ],
                    flag_groups = [flag_group(flags = [
                        "-fembed-bitcode",
                        "-Xlinker",
                        "-bitcode_verify",
                        "-Xlinker",
                        "-bitcode_hide_symbols",
                        "-Xlinker",
                        "-bitcode_symbol_map",
                        "-Xlinker",
                        "BITCODE_TOUCH_SYMBOL_MAP=%{bitcode_symbol_map_path}",
                    ])],
                ),
            ],
        )

        bitcode_embedded_markers_feature = feature(
            name = "bitcode_embedded_markers",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.objc_compile,
                        ACTION_NAMES.objcpp_compile,
                    ],
                    flag_groups = [flag_group(flags = ["-fembed-bitcode-marker"])],
                ),
                flag_set(
                    actions = all_link_actions + [
                        ACTION_NAMES.objc_executable,
                        ACTION_NAMES.objcpp_executable,
                    ],
                    flag_groups = [flag_group(flags = ["-fembed-bitcode-marker"])],
                ),
            ],
        )
    else:
        bitcode_embedded_markers_feature = feature(name = "bitcode_embedded_markers")
        bitcode_embedded_feature = feature(name = "bitcode_embedded")

    shared_features = [
        fastbuild_feature,
        no_legacy_features_feature,
        opt_feature,
        dbg_feature,
        link_libcpp_feature,
        compile_all_modules_feature,
        exclude_private_headers_in_module_maps_feature,
        has_configured_linker_path_feature,
        only_doth_headers_in_module_maps_feature,
        default_compile_flags_feature,
        debug_prefix_map_pwd_is_dot_feature,
        generate_dsym_file_feature,
        contains_objc_source_feature,
        objc_actions_feature,
        strip_debug_symbols_feature,
        symbol_counts_feature,
        shared_flag_feature,
        kernel_extension_feature,
        linkstamps_feature,
        output_execpath_flags_feature,
        archiver_flags_feature,
        runtime_root_flags_feature,
        input_param_flags_feature,
        force_pic_flags_feature,
        pch_feature,
        module_maps_feature,
        use_objc_modules_feature,
        no_enable_modules_feature,
        apply_default_warnings_feature,
        includes_feature,
        include_paths_feature,
        sysroot_feature,
        dependency_file_feature,
        pic_feature,
        per_object_debug_info_feature,
        preprocessor_defines_feature,
        framework_paths_feature,
        random_seed_feature,
        fdo_instrument_feature,
        fdo_optimize_feature,
        autofdo_feature,
        lipo_feature,
        coverage_feature,
        llvm_coverage_map_format_feature,
        gcc_coverage_map_format_feature,
        apply_default_compiler_flags_feature,
        include_system_dirs_feature,
        headerpad_feature,
        bitcode_embedded_feature,
        bitcode_embedded_markers_feature,
        objc_arc_feature,
        no_objc_arc_feature,
        apple_env_feature,
        user_link_flags_feature,
        default_link_flags_feature,
        version_min_feature,
        dead_strip_feature,
        cpp_linker_flags_feature,
        apply_implicit_frameworks_feature,
        link_cocoa_feature,
        apply_simulator_compiler_flags_feature,
        unfiltered_cxx_flags_feature,
        user_compile_flags_feature,
        unfiltered_compile_flags_feature,
        linker_param_file_feature,
        compiler_input_flags_feature,
        compiler_output_flags_feature,
        objcopy_embed_flags_feature,
    ]

    if ctx.attr.cpu == "armeabi-v7a":
        custom_features = [
            supports_pic_feature,
        ]
    elif ctx.attr.cpu == "darwin_x86_64":
        custom_features = [
            supports_dynamic_linker_feature,
            dynamic_linking_mode_feature,
        ]
    else:
        custom_features = []

    cxx_builtin_include_directories = [
%{cxx_builtin_include_directories}
    ]

    make_variables = [
        make_variable(
            name = "STACK_FRAME_UNLIMITED",
            value = "-Wframe-larger-than=100000000 -Wno-vla",
        ),
    ]

    if (ctx.attr.cpu == "armeabi-v7a"):
        tool_paths = [
            tool_path(name = "ar", path = "/bin/false"),
            tool_path(name = "compat-ld", path = "/bin/false"),
            tool_path(name = "cpp", path = "/bin/false"),
            tool_path(name = "dwp", path = "/bin/false"),
            tool_path(name = "gcc", path = "/bin/false"),
            tool_path(name = "gcov", path = "/bin/false"),
            tool_path(name = "ld", path = "/bin/false"),
            tool_path(name = "nm", path = "/bin/false"),
            tool_path(name = "objcopy", path = "/bin/false"),
            tool_path(name = "objdump", path = "/bin/false"),
            tool_path(name = "strip", path = "/bin/false"),
        ]
    else:
        tool_paths = [
            tool_path(name = "ar", path = "wrapped_ar"),
            tool_path(name = "compat-ld", path = "/usr/bin/ld"),
            tool_path(name = "cpp", path = "/usr/bin/cpp"),
            tool_path(name = "dwp", path = "/usr/bin/dwp"),
            tool_path(name = "gcc", path = "cc_wrapper.sh"),
            tool_path(name = "gcov", path = "/usr/bin/gcov"),
            tool_path(name = "ld", path = "/usr/bin/ld"),
            tool_path(name = "nm", path = "/usr/bin/nm"),
            tool_path(name = "objcopy", path = "/usr/bin/objcopy"),
            tool_path(name = "objdump", path = "/usr/bin/objdump"),
            tool_path(name = "strip", path = "/usr/bin/strip"),
        ]


    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(out, "Fake executable")
    return [
        cc_common.create_cc_toolchain_config_info(
            ctx = ctx,
            features = shared_features + custom_features,
            action_configs = action_configs,
            artifact_name_patterns = [],
            cxx_builtin_include_directories = cxx_builtin_include_directories,
            toolchain_identifier = toolchain_identifier,
            host_system_name = host_system_name,
            target_system_name = target_system_name,
            target_cpu = ctx.attr.cpu,
            target_libc = target_libc,
            compiler = "compiler",
            abi_version = abi_version,
            abi_libc_version = abi_libc_version,
            tool_paths = tool_paths,
            make_variables = make_variables,
            builtin_sysroot = None,
            cc_target_os = "apple"
        ),
        DefaultInfo(
            executable = out,
        ),
    ]

cc_toolchain_config =  rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory=True),
        "compiler": attr.string(),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
