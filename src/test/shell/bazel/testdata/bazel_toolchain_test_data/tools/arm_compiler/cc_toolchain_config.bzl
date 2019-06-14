#
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
#
# Tests compiling using an external Linaro toolchain on a Linux machine
#

"""Implementation of a rule that configures a Linaro toolchain."""

load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "feature",
    "flag_group",
    "flag_set",
    "tool",
    "tool_path",
    "with_feature_set",
)
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

def _impl(ctx):
    if (ctx.attr.cpu == "armeabi-v7a"):
        toolchain_identifier = "armeabi-v7a"
    elif (ctx.attr.cpu == "k8"):
        toolchain_identifier = "local"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "armeabi-v7a"):
        host_system_name = "armeabi-v7a"
    elif (ctx.attr.cpu == "k8"):
        host_system_name = "local"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "armeabi-v7a"):
        target_system_name = "arm_a15"
    elif (ctx.attr.cpu == "k8"):
        target_system_name = "local"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "armeabi-v7a"):
        target_cpu = "armeabi-v7a"
    elif (ctx.attr.cpu == "k8"):
        target_cpu = "k8"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "armeabi-v7a"):
        target_libc = "glibc_2.19"
    elif (ctx.attr.cpu == "k8"):
        target_libc = "local"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "k8"):
        compiler = "compiler"
    elif (ctx.attr.cpu == "armeabi-v7a"):
        compiler = "gcc"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "armeabi-v7a"):
        abi_version = "gcc"
    elif (ctx.attr.cpu == "k8"):
        abi_version = "local"
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "armeabi-v7a"):
        abi_libc_version = "glibc_2.19"
    elif (ctx.attr.cpu == "k8"):
        abi_libc_version = "local"
    else:
        fail("Unreachable")

    cc_target_os = None

    builtin_sysroot = None

    all_link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

    if (ctx.attr.cpu == "armeabi-v7a"):
        objcopy_embed_data_action = action_config(
            action_name = "objcopy_embed_data",
            enabled = True,
            tools = [
                tool(path = "linaro_linux_gcc/arm-linux-gnueabihf-objcopy"),
            ],
        )
    elif (ctx.attr.cpu == "k8"):
        objcopy_embed_data_action = action_config(
            action_name = "objcopy_embed_data",
            enabled = True,
            tools = [tool(path = "/usr/bin/objcopy")],
        )
    else:
        objcopy_embed_data_action = None

    action_configs = [objcopy_embed_data_action]

    if (ctx.attr.cpu == "k8"):
        unfiltered_compile_flags_feature = feature(
            name = "unfiltered_compile_flags",
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
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-fno-canonical-system-headers",
                                "-Wno-builtin-macro-redefined",
                                "-D__DATE__=\"redacted\"",
                                "-D__TIMESTAMP__=\"redacted\"",
                                "-D__TIME__=\"redacted\"",
                            ],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "armeabi-v7a"):
        unfiltered_compile_flags_feature = feature(
            name = "unfiltered_compile_flags",
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
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-no-canonical-prefixes",
                                "-Wno-builtin-macro-redefined",
                                "-D__DATE__=\"redacted\"",
                                "-D__TIMESTAMP__=\"redacted\"",
                                "-D__TIME__=\"redacted\"",
                            ],
                        ),
                    ],
                ),
            ],
        )
    else:
        unfiltered_compile_flags_feature = None

    supports_dynamic_linker_feature = feature(name = "supports_dynamic_linker", enabled = True)

    if (ctx.attr.cpu == "armeabi-v7a"):
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
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "--sysroot=external/org_linaro_components_toolchain_gcc_5_3_1/arm-linux-gnueabihf/libc",
                                "-mfloat-abi=hard",
                                "-nostdinc",
                                "-isystem",
                                "external/org_linaro_components_toolchain_gcc_5_3_1/lib/gcc/arm-linux-gnueabihf/5.3.1/include",
                                "-isystem",
                                "external/org_linaro_components_toolchain_gcc_5_3_1/arm-linux-gnueabihf/libc/usr/include",
                                "-isystem",
                                "external/org_linaro_components_toolchain_gcc_5_3_1/lib/gcc/arm-linux-gnueabihf/5.3.1/include-fixed",
                                "-isystem",
                                "external/org_linaro_components_toolchain_gcc_5_3_1/arm-linux-gnueabihf/libc/usr/include",
                                "-U_FORTIFY_SOURCE",
                                "-fstack-protector",
                                "-fPIE",
                                "-fdiagnostics-color=always",
                                "-Wall",
                                "-Wunused-but-set-parameter",
                                "-Wno-free-nonheap-object",
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
                    ],
                    flag_groups = [flag_group(flags = ["-g"])],
                    with_features = [with_feature_set(features = ["dbg"])],
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
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-g0",
                                "-O2",
                                "-DNDEBUG",
                                "-ffunction-sections",
                                "-fdata-sections",
                            ],
                        ),
                    ],
                    with_features = [with_feature_set(features = ["opt"])],
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
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-isystem",
                                "external/org_linaro_components_toolchain_gcc_5_3_1/arm-linux-gnueabihf/include/c++/5.3.1/arm-linux-gnueabihf",
                                "-isystem",
                                "external/org_linaro_components_toolchain_gcc_5_3_1/arm-linux-gnueabihf/include/c++/5.3.1",
                                "-isystem",
                                "external/org_linaro_components_toolchain_gcc_5_3_1/include/c++/5.3.1/arm-linux-gnueabihf",
                                "-isystem",
                                "external/org_linaro_components_toolchain_gcc_5_3_1/include/c++/5.3.1",
                            ],
                        ),
                    ],
                ),
            ],
        )
    elif (ctx.attr.cpu == "k8"):
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
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-U_FORTIFY_SOURCE",
                                "-D_FORTIFY_SOURCE=2",
                                "-fstack-protector",
                                "-Wall",
                                "-Wl,-z,-relro,-z,now",
                                "-Wunused-but-set-parameter",
                                "-Wno-free-nonheap-object",
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
                    ],
                    flag_groups = [flag_group(flags = ["-g"])],
                    with_features = [with_feature_set(features = ["dbg"])],
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
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-g0",
                                "-O2",
                                "-DNDEBUG",
                                "-ffunction-sections",
                                "-fdata-sections",
                            ],
                        ),
                    ],
                    with_features = [with_feature_set(features = ["opt"])],
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
                    flag_groups = [flag_group(flags = ["-std=c++0x"])],
                ),
            ],
        )
    else:
        default_compile_flags_feature = None

    supports_pic_feature = feature(name = "supports_pic", enabled = True)

    opt_feature = feature(name = "opt")

    user_compile_flags_feature = feature(
        name = "user_compile_flags",
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

    sysroot_feature = feature(
        name = "sysroot",
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
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
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

    if (ctx.attr.cpu == "armeabi-v7a"):
        default_link_flags_feature = feature(
            name = "default_link_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [
                        flag_group(
                            flags = [
                                "--sysroot=external/org_linaro_components_toolchain_gcc_5_3_1/arm-linux-gnueabihf/libc",
                                "-lstdc++",
                                "-latomic",
                                "-lm",
                                "-lpthread",
                                "-Ltools/arm_compiler/linaro_linux_gcc/clang_more_libs",
                                "-Lexternal/org_linaro_components_toolchain_gcc_5_3_1/arm-linux-gnueabihf/lib",
                                "-Lexternal/org_linaro_components_toolchain_gcc_5_3_1/arm-linux-gnueabihf/libc/lib",
                                "-Lexternal/org_linaro_components_toolchain_gcc_5_3_1/arm-linux-gnueabihf/libc/usr/lib",
                                "-Bexternal/org_linaro_components_toolchain_gcc_5_3_1/arm-linux-gnueabihf/bin",
                                "-Wl,--dynamic-linker=/lib/ld-linux-armhf.so.3",
                                "-no-canonical-prefixes",
                                "-pie",
                                "-Wl,-z,relro,-z,now",
                            ],
                        ),
                    ],
                ),
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = ["-Wl,--gc-sections"])],
                    with_features = [with_feature_set(features = ["opt"])],
                ),
            ],
        )
    elif (ctx.attr.cpu == "k8"):
        default_link_flags_feature = feature(
            name = "default_link_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-lstdc++",
                                "-lm",
                                "-Wl,-no-as-needed",
                                "-pass-exit-codes",
                            ],
                        ),
                    ],
                ),
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = ["-Wl,--gc-sections"])],
                    with_features = [with_feature_set(features = ["opt"])],
                ),
            ],
        )
    else:
        default_link_flags_feature = None

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

    dbg_feature = feature(name = "dbg")

    if (ctx.attr.cpu == "k8"):
        features = [
            default_compile_flags_feature,
            default_link_flags_feature,
            supports_dynamic_linker_feature,
            supports_pic_feature,
            objcopy_embed_flags_feature,
            opt_feature,
            dbg_feature,
            user_compile_flags_feature,
            sysroot_feature,
            unfiltered_compile_flags_feature,
        ]
    elif (ctx.attr.cpu == "armeabi-v7a"):
        features = [
            default_compile_flags_feature,
            default_link_flags_feature,
            supports_pic_feature,
            objcopy_embed_flags_feature,
            opt_feature,
            dbg_feature,
            user_compile_flags_feature,
            sysroot_feature,
            unfiltered_compile_flags_feature,
        ]
    else:
        fail("Unreachable")

    if (ctx.attr.cpu == "armeabi-v7a"):
        cxx_builtin_include_directories = [
            "%package(@org_linaro_components_toolchain_gcc_5_3_1//include)%",
            "%package(@org_linaro_components_toolchain_gcc_5_3_1//arm-linux-gnueabihf/libc/usr/include)%",
            "%package(@org_linaro_components_toolchain_gcc_5_3_1//arm-linux-gnueabihf/libc/usr/lib/include)%",
            "%package(@org_linaro_components_toolchain_gcc_5_3_1//arm-linux-gnueabihf/libc/lib/gcc/arm-linux-gnueabihf/5.3.1/include-fixed)%",
            "%package(@org_linaro_components_toolchain_gcc_5_3_1//include)%/c++/5.3.1",
            "%package(@org_linaro_components_toolchain_gcc_5_3_1//arm-linux-gnueabihf/libc/lib/gcc/arm-linux-gnueabihf/5.3.1/include)%",
            "%package(@org_linaro_components_toolchain_gcc_5_3_1//arm-linux-gnueabihf/libc/lib/gcc/arm-linux-gnueabihf/5.3.1/include-fixed)%",
            "%package(@org_linaro_components_toolchain_gcc_5_3_1//lib/gcc/arm-linux-gnueabihf/5.3.1/include)%",
            "%package(@org_linaro_components_toolchain_gcc_5_3_1//lib/gcc/arm-linux-gnueabihf/5.3.1/include-fixed)%",
            "%package(@org_linaro_components_toolchain_gcc_5_3_1//arm-linux-gnueabihf/include)%/c++/5.3.1",
        ]
    elif (ctx.attr.cpu == "k8"):
        cxx_builtin_include_directories = [
            "/usr/include/c++/4.8",
            "/usr/include/x86_64-linux-gnu/c++/4.8",
            "/usr/include/c++/4.8/backward",
            "/usr/lib/gcc/x86_64-linux-gnu/4.8/include",
            "/usr/local/include",
            "/usr/lib/gcc/x86_64-linux-gnu/4.8/include-fixed",
            "/usr/include/x86_64-linux-gnu",
            "/usr/include",
        ]
    else:
        fail("Unreachable")

    artifact_name_patterns = []

    make_variables = []

    if (ctx.attr.cpu == "armeabi-v7a"):
        tool_paths = [
            tool_path(
                name = "ar",
                path = "linaro_linux_gcc/arm-linux-gnueabihf-ar",
            ),
            tool_path(
                name = "compat-ld",
                path = "linaro_linux_gcc/arm-linux-gnueabihf-ld",
            ),
            tool_path(
                name = "cpp",
                path = "linaro_linux_gcc/arm-linux-gnueabihf-gcc",
            ),
            tool_path(
                name = "dwp",
                path = "linaro_linux_gcc/arm-linux-gnueabihf-dwp",
            ),
            tool_path(
                name = "gcc",
                path = "linaro_linux_gcc/arm-linux-gnueabihf-gcc",
            ),
            tool_path(
                name = "gcov",
                path = "arm-frc-linux-gnueabi/arm-frc-linux-gnueabi-gcov-4.9",
            ),
            tool_path(
                name = "ld",
                path = "linaro_linux_gcc/arm-linux-gnueabihf-ld",
            ),
            tool_path(
                name = "nm",
                path = "linaro_linux_gcc/arm-linux-gnueabihf-nm",
            ),
            tool_path(
                name = "objcopy",
                path = "linaro_linux_gcc/arm-linux-gnueabihf-objcopy",
            ),
            tool_path(
                name = "objdump",
                path = "linaro_linux_gcc/arm-linux-gnueabihf-objdump",
            ),
            tool_path(
                name = "strip",
                path = "linaro_linux_gcc/arm-linux-gnueabihf-strip",
            ),
        ]
    elif (ctx.attr.cpu == "k8"):
        tool_paths = [
            tool_path(name = "ar", path = "/usr/bin/ar"),
            tool_path(name = "cpp", path = "/usr/bin/cpp"),
            tool_path(name = "dwp", path = "/usr/bin/dwp"),
            tool_path(name = "gcc", path = "/usr/bin/gcc"),
            tool_path(name = "gcov", path = "/usr/bin/gcov"),
            tool_path(name = "ld", path = "/usr/bin/ld"),
            tool_path(name = "nm", path = "/usr/bin/nm"),
            tool_path(name = "objcopy", path = "/usr/bin/objcopy"),
            tool_path(name = "objdump", path = "/usr/bin/objdump"),
            tool_path(name = "strip", path = "/usr/bin/strip"),
        ]
    else:
        fail("Unreachable")

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
        "cpu": attr.string(mandatory = True, values = ["armeabi-v7a", "k8"]),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
