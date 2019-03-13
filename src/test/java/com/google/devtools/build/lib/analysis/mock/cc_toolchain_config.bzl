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
    "action_config",
    "artifact_name_pattern",
    "env_entry",
    "env_set",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "tool",
    "tool_path",
    "with_feature_set",
)
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

_FEATURE_NAMES = struct(
    no_legacy_features = "no_legacy_features",
    do_not_split_linking_cmdline = "do_not_split_linking_cmdline",
    supports_dynamic_linker = "supports_dynamic_linker",
    supports_interface_shared_libraries = "supports_interface_shared_libraries",
    pic = "pic",
    parse_headers = "parse_headers",
    layering_check = "layering_check",
    header_modules = "header_modules",
    header_module_compile = "header_module_compile",
    header_module_codegen = "header_module_codegen",
    module_maps = "module_maps",
    use_header_modules = "use_header_modules",
    module_map_home_cwd = "module_map_home_cwd",
    env_feature = "env_feature",
    static_env_feature = "static_env_feature",
    host = "host",
    nonhost = "nonhost",
    user_compile_flags = "user_compile_flags",
    thin_lto = "thin_lto",
    thin_lto_linkstatic_tests_use_shared_nonlto_backends = "thin_lto_linkstatic_tests_use_shared_nonlto_backends",
    thin_lto_all_linkstatic_use_shared_nonlto_backends = "thin_lto_all_linkstatic_use_shared_nonlto_backends",
    enable_afdo_thinlto = "enable_afdo_thinlto",
    enable_fdo_thinlto = "enable_fdo_thinlto",
    xbinaryfdo_implicit_thinlto = "xbinaryfdo_implicit_thinlto",
    enable_xbinaryfdo_thinlto = "enable_xbinaryfdo_thinlto",
    autofdo = "autofdo",
    is_cc_fake_binary = "is_cc_fake_binary",
    xbinaryfdo = "xbinaryfdo",
    fdo_optimize = "fdo_optimize",
    fdo_implicit_thinlto = "fdo_implicit_thinlto",
    fdo_instrument = "fdo_instrument",
    supports_pic = "supports_pic",
    copy_dynamic_libraries_to_binary = "copy_dynamic_libraries_to_binary",
    per_object_debug_info = "per_object_debug_info",
    supports_start_end_lib = "supports_start_end_lib",
    targets_windows = "targets_windows",
)

_no_legacy_features_feature = feature(name = _FEATURE_NAMES.no_legacy_features)

_do_not_split_linking_cmdline_feature = feature(name = _FEATURE_NAMES.do_not_split_linking_cmdline)

_supports_dynamic_linker_feature = feature(
    name = _FEATURE_NAMES.supports_dynamic_linker,
    enabled = True,
)

_supports_interface_shared_libraries_feature = feature(
    name = _FEATURE_NAMES.supports_interface_shared_libraries,
    enabled = True,
)

_pic_feature = feature(
    name = _FEATURE_NAMES.pic,
    enabled = True,
    flag_sets = [
        flag_set(
            actions = [
                ACTION_NAMES.assemble,
                ACTION_NAMES.preprocess_assemble,
                ACTION_NAMES.linkstamp_compile,
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.cpp_module_codegen,
                ACTION_NAMES.cpp_module_compile,
            ],
            flag_groups = [
                flag_group(
                    expand_if_available = "pic",
                    flags = ["-fPIC"],
                ),
            ],
        ),
    ],
)

_parse_headers_feature = feature(
    name = _FEATURE_NAMES.parse_headers,
    flag_sets = [
        flag_set(
            actions = [ACTION_NAMES.cpp_header_parsing],
            flag_groups = [
                flag_group(
                    flags = ["<c++-header-parsing>"],
                ),
            ],
        ),
    ],
)

_layering_check_feature = feature(
    name = FEATURE_NAMES.layering_check,
    flag_sets = [
        flag_set(
            actions = [
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.cpp_header_parsing,
                ACTION_NAMES.cpp_module_compile,
            ],
            flag_groups = [
                flag_group(
                    iterate_over = "dependent_module_map_files",
                    flags = [
                        "dependent_module_map_file:%{dependent_module_map_files}",
                    ],
                ),
            ],
        ),
    ],
)

_header_modules_feature = feature(
    name = _FEATURE_NAMES.header_modules,
    implies = ["use_header_modules", "header_module_compile"],
)

_header_module_compile_feature = feature(
    name = _FEATURE_NAMES.header_module_compile,
    enabled = True,
    implies = ["module_maps"],
    flag_sets = [
        flag_set(
            actions = [ACTION_NAMES.cpp_module_compile],
            flag_groups = [
                flag_group(
                    flags = ["--woohoo_modules"],
                ),
            ],
        ),
        flag_set(
            actions = [ACTION_NAMES.cpp_module_codegen],
            flag_groups = [
                flag_group(
                    flags = ["--this_is_modules_codegen"],
                ),
            ],
        ),
    ],
)

_header_module_codegen_feature = feature(
    name = _FEATURE_NAMES.header_module_codegen,
    implies = ["header_modules"],
)

_module_maps_feature = feature(
    name = _FEATURE_NAMES.module_maps,
    enabled = True,
    flag_sets = [
        flag_set(
            actions = [
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.cpp_header_parsing,
                ACTION_NAMES.cpp_module_compile,
            ],
            flag_groups = [
                flag_group(
                    flags = [
                        "module_name:%{module_name}",
                        "module_map_file:%{module_map_file}",
                    ],
                ),
            ],
        ),
    ],
)

_use_header_modules_feature = feature(
    name = _FEATURE_NAMES.use_header_modules,
    flag_sets = [
        flag_set(
            actions = [
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.cpp_header_parsing,
                ACTION_NAMES.cpp_module_compile,
            ],
            flag_groups = [
                flag_group(
                    iterate_over = "module_files",
                    flags = ["module_file:%{module_files}"],
                ),
            ],
        ),
    ],
)

_header_modules_feature_configuration = [
    _header_modules_feature,
    _header_module_compile_feature,
    _header_module_codegen_feature,
    _module_maps_feature,
    _use_header_modules_feature,
]

_module_map_home_cwd_feature = feature(
    name = _FEATURE_NAMES.module_map_home_cwd,
    enabled = True,
    flag_sets = [
        flag_set(
            actions = [
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.cpp_header_parsing,
                ACTION_NAMES.cpp_module_compile,
                ACTION_NAMES.cpp_header_parsing,
                ACTION_NAMES.preprocess_assemble,
            ],
            flag_groups = [
                flag_group(
                    flags = ["<flag>"],
                ),
            ],
        ),
    ],
)

_env_feature = feature(
    name = _FEATURE_NAMES.env_feature,
    implies = ["static_env_feature", "module_maps"],
)

_static_env_feature = feature(
    name = _FEATURE_NAMES.static_env_feature,
    env_sets = [
        env_set(
            actions = [
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.cpp_header_parsing,
                ACTION_NAMES.cpp_module_compile,
            ],
            env_entries = [
                env_entry(
                    key = "cat",
                    value = "meow",
                ),
            ],
        ),
    ],
)

_module_maps_env_var_feature = feature(
    name = _FEATURE_NAMES.module_maps,
    enabled = True,
    env_sets = [
        env_set(
            actions = [
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.cpp_header_parsing,
                ACTION_NAMES.cpp_module_compile,
            ],
            env_entries = [
                env_entry(
                    key = "module",
                    value = "module_name:%{module_name}",
                ),
            ],
        ),
    ],
)

_env_var_feature_configuration = [
    _env_feature,
    _static_env_feature,
    _module_maps_env_var_feature,
]

_host_feature = feature(
    name = _FEATURE_NAMES.host,
    flag_sets = [
        flag_set(
            actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
            flag_groups = [flag_group(flags = ["-host"])],
        ),
    ],
)

_nonhost_feature = feature(
    name = _FEATURE_NAMES.nonhost,
    flag_sets = [
        flag_set(
            actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
            flag_groups = [flag_group(flags = ["-nonhost"])],
        ),
    ],
)

_host_and_nonhost_configuration = [
    _host_feature,
    _nonhost_feature,
]

_user_compile_flags_feature = feature(
    name = _FEATURE_NAMES.user_compile_flags,
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

_thin_lto_feature = feature(
    name = _FEATURE_NAMES.thin_lto,
    flag_sets = [
        flag_set(
            actions = [
                ACTION_NAMES.cpp_link_executable,
                ACTION_NAMES.cpp_link_dynamic_library,
                ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ACTION_NAMES.cpp_link_static_library,
            ],
            flag_groups = [
                flag_group(
                    flags = ["thinlto_param_file=%{thinlto_param_file}"],
                ),
            ],
        ),
        flag_set(
            actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
            flag_groups = [
                flag_group(flags = ["-flto=thin"]),
                flag_group(
                    flags = ["lto_indexing_bitcode=%{lto_indexing_bitcode_file}"],
                    expand_if_available = "lto_indexing_bitcode_file",
                ),
            ],
        ),
        flag_set(
            actions = [ACTION_NAMES.lto_indexing],
            flag_groups = [
                flag_group(
                    flags = [
                        "param_file=%{thinlto_indexing_param_file}",
                        "prefix_replace=%{thinlto_prefix_replace}",
                    ],
                ),
                flag_group(
                    flags = ["object_suffix_replace=%{thinlto_object_suffix_replace}"],
                    expand_if_available = "thinlto_object_suffix_replace",
                ),
                flag_group(
                    flags = ["thinlto_merged_object_file=%{thinlto_merged_object_file}"],
                    expand_if_available = "thinlto_merged_object_file",
                ),
            ],
        ),
        flag_set(
            actions = [ACTION_NAMES.lto_backend],
            flag_groups = [
                flag_group(
                    flags = [
                        "thinlto_index=%{thinlto_index}",
                        "thinlto_output_object_file=%{thinlto_output_object_file}",
                        "thinlto_input_bitcode_file=%{thinlto_input_bitcode_file}",
                    ],
                ),
            ],
        ),
    ],
    requires = [feature_set(features = ["nonhost"])],
)

_thin_lto_linkstatic_tests_use_shared_nonlto_backends_feature = feature(
    name = _FEATURE_NAMES.thin_lto_linkstatic_tests_use_shared_nonlto_backends,
)

_thin_lto_all_linkstatic_use_shared_nonlto_backends_feature = feature(
    name = _FEATURE_NAMES.thin_lto_all_linkstatic_use_shared_nonlto_backends,
)

_enable_afdo_thinlto_feature = feature(
    name = _FEATURE_NAMES.enable_afdo_thinlto,
    requires = [feature_set(features = ["autofdo_implicit_thinlto"])],
    implies = ["thin_lto"],
)

_autofdo_implicit_thinlto_feature = feature(name = "autofdo_implicit_thinlto")

_enable_fdo_thin_lto_feature = feature(
    name = _FEATURE_NAMES.enable_fdo_thinlto,
    requires = [feature_set(features = ["fdo_implicit_thinlto"])],
    implies = ["thin_lto"],
)

_fdo_implicit_thinlto_feature = feature(name = _FEATURE_NAMES.fdo_implicit_thinlto)

_enable_xbinaryfdo_thinlto_feature = feature(
    name = _FEATURE_NAMES.enable_xbinaryfdo_thinlto,
    requires = [feature_set(features = ["xbinaryfdo_implicit_thinlto"])],
    implies = ["thin_lto"],
)

_xbinaryfdo_implicit_thinlto_feature = feature(name = _FEATURE_NAMES.xbinaryfdo_implicit_thinlto)

_autofdo_feature = feature(
    name = _FEATURE_NAMES.autofdo,
    flag_sets = [
        flag_set(
            actions = [
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.lto_backend,
            ],
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

_is_cc_fake_binary_feature = feature(name = _FEATURE_NAMES.is_cc_fake_binary)

_xbinaryfdo_feature = feature(
    name = _FEATURE_NAMES.xbinaryfdo,
    flag_sets = [
        flag_set(
            actions = [
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.lto_backend,
            ],
            flag_groups = [
                flag_group(
                    flags = [
                        "-fauto-profile=%{fdo_profile_path}",
                        "-fprofile-correction",
                    ],
                ),
            ],
            with_features = [
                with_feature_set(not_features = ["is_cc_fake_binary"]),
            ],
        ),
    ],
    provides = ["profile"],
)

_fdo_optimize_feature = feature(
    name = _FEATURE_NAMES.fdo_optimize,
    flag_sets = [
        flag_set(
            actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
            flag_groups = [
                flag_group(
                    flags = [
                        "-fprofile-use=%{fdo_profile_path}",
                        "-Xclang-only=-Wno-profile-instr-unprofiled",
                        "-Xclang-only=-Wno-profile-instr-out-of-date",
                        "-Xclang-only=-Wno-backend-plugin",
                        "-fprofile-correction",
                    ],
                    expand_if_available = "fdo_profile_path",
                ),
            ],
        ),
    ],
    provides = ["profile"],
)

_fdo_instrument_feature = feature(
    name = _FEATURE_NAMES.fdo_instrument,
    flag_sets = [
        flag_set(
            actions = [
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.cpp_link_dynamic_library,
                ACTION_NAMES.link_nodeps_dynamic_library,
                ACTION_NAMES.cpp_link_executable,
            ],
            flag_groups = [
                flag_group(
                    flags = ["fdo_instrument_option", "path=%{fdo_instrument_path}"],
                ),
            ],
        ),
    ],
    provides = ["profile"],
)

_per_object_debug_info_feature = feature(
    name = _FEATURE_NAMES.per_object_debug_info,
    enabled = True,
    flag_sets = [
        flag_set(
            actions = [
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.assemble,
                ACTION_NAMES.preprocess_assemble,
                ACTION_NAMES.cpp_module_codegen,
                ACTION_NAMES.lto_backend,
            ],
            flag_groups = [
                flag_group(
                    flags = ["per_object_debug_info_option"],
                    expand_if_available = "per_object_debug_info_file",
                ),
            ],
        ),
    ],
)

_copy_dynamic_libraries_to_binary_feature = feature(
    name = _FEATURE_NAMES.copy_dynamic_libraries_to_binary,
)

_supports_start_end_lib_feature = feature(
    name = _FEATURE_NAMES.supports_start_end_lib,
    enabled = True,
)

_supports_pic_feature = feature(name = _FEATURE_NAMES.supports_pic, enabled = True)

_targets_windows_feature = feature(
    name = _FEATURE_NAMES.targets_windows,
    enabled = True,
    implies = ["copy_dynamic_libraries_to_binary"],
)

_static_link_cpp_runtimes_feature = feature(
    name = _FEATURE_NAMES.static_link_cpp_runtimes,
    enabled = True,
)

_feature_name_to_feature = {
    _FEATURE_NAMES.no_legacy_features: _no_legacy_features_feature,
    _FEATURE_NAMES.do_not_split_linking_cmdline: _do_not_split_linking_cmdline_feature,
    _FEATURE_NAMES.supports_dynamic_linker: _supports_dynamic_linker_feature,
    _FEATURE_NAMES.supports_interface_shared_libraries: _supports_interface_shared_libraries_feature,
    _FEATURE_NAMES.pic: _pic_feature,
    _FEATURE_NAMES.parse_headers: _parse_headers_feature,
    _FEATURE_NAMES.layering_check: _layering_check_feature,
    _FEATURE_NAMES.module_map_home_cwd: _module_map_home_cwd_feature,
    _FEATURE_NAMES.user_compile_flags: _user_compile_flags_feature,
    _FEATURE_NAMES.thin_lto_feature: _thin_lto_feature,
    _FEATURE_NAMES.thin_lto_linkstatic_tests_use_shared_nonlto_backend: _thin_lto_linkstatic_tests_use_shared_nonlto_backends_feature,
    _FEATURE_NAMES.thin_lto_all_linkstatic_use_shared_nonlto_backends: _thin_lto_all_linkstatic_use_shared_nonlto_backends_feature,
    _FEATURE_NAMES.enable_afdo_thin_lto: _enable_afdo_thinlto_feature,
    _FEATURE_NAMES.autofdo_implicit_thinlto: _autofdo_implicit_thinlto_feature,
    _FEATURE_NAMES.enable_fdo_thinlto: _enable_fdo_thin_lto_feature,
    _FEATURE_NAMES.fdo_implicit_thinlto: _fdo_implicit_thinlto_feature,
    _FEATURE_NAMES.enable_xbinaryfdo_thinlto: _enable_xbinaryfdo_thinlto_feature,
    _FEATURE_NAMES.xbinaryfdo_implicit_thinlto: _xbinaryfdo_implicit_thinlto_feature,
    _FEATURE_NAMES.autofdo: _autofdo_feature,
    _FEATURE_NAMES.is_cc_fake_binary: _is_cc_fake_binary_feature,
    _FEATURE_NAMES.xbinaryfdo: _xbinaryfdo_feature,
    _FEATURE_NAMES.fdo_optimize: _fdo_optimize_feature,
    _FEATURE_NAMES.fdo_instrument_feature: _fdo_instrument_feature,
    _FEATURE_NAMES.per_object_debug_info: _per_object_debug_info_feature,
    _FEATURE_NAMES.copy_dynamic_libraries_to_binary: _copy_dynamic_libraries_to_binary_feature,
    _FEATURE_NAMES.supports_start_end_lib: _supports_start_end_lib_feature,
    _FEATURE_NAMES.supports_pic: _supports_pic_feature,
    _FEATURE_NAMES.targets_windows: _targets_windows_feature,
    _FEATURE_NAMES.module_maps: _module_maps_feature,
    _FEATURE_NAMES.static_link_cpp_runtimes: _static_link_cpp_runtimes_feature,
    "header_modules_feature_configuration": _header_modules_feature_configuration,
    "env_var_feature_configuration": _env_var_feature_configuration,
    "host_and_nonhost_configuration": _host_and_nonhost_configuration,
}

_static_link_as_dot_lib_pattern = artifact_name_pattern(
    category_name = "static_library",
    prefix = "lib",
    extension = ".lib",
)

_static_link_as_dot_a_pattern = artifact_name_pattern(
    category_name = "static_library",
    prefix = "lib",
    extension = ".a",
)

_artifact_name_to_artifact_pattern = {
    "static_link_as_dot_lib": _static_link_as_dot_lib_pattern,
    "static_link_as_dot_a": _static_link_as_dot_a_pattern,
}

def _get_features_for_configuration(name):
    f = _feature_name_to_feature[name]
    if f == None:
        fail("Feature not defined: " + name)
    if type(f) == type([]):
        return f
    else:
        return [f]

def _get_action_config(name):
    return action_config(
        action_name = name,
        tools = [tool(tool_path = "DUMMY_TOOL")],
    )

def _get_artifact_name_pattern(name):
    artifact = _artifact_name_to_artifact_pattern[name]
    if artifact == None:
        fail("Artifact name pattern not defined: " + name)
    return artifact

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

    all_link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

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

    for name in ctx.attr.features:
        features.extend(_get_features_for_configuration(name))

    cxx_builtin_include_directories = ["/usr/lib/gcc/", "/usr/local/include", "/usr/include"]

    artifact_name_patterns = []

    for name in ctx.attr.artifact_name_patterns:
        artifact_name_patterns.append(_get_artifact_name_pattern(name))

    action_configs = []

    for name in ctx.attr.action_configs:
        action_configs.append(_get_action_config(name))

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
        "cpu": attr.string(default = "k8"),
        "compiler": attr.string(default = "compiler"),
        "toolchain_identifier": attr.string(default = "mock-llvm-toolchain-k8"),
        "host_system_name": attr.string(default = "local"),
        "target_system_name": attr.string(default = "local"),
        "target_libc": attr.string(default = "local"),
        "abi_version": attr.string(default = "local"),
        "abi_libc_version": attr.string(default = "local"),
        "features": attr.string_list(),
        "action_configs": attr.string_list(),
        "artifact_name_patterns": attr.string_list(),
        "cc_target_os": attr.string(),
        "builtin_sysroot": attr.string(default = "/usr/grte/v1"),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
