# Copyright 2025 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# LINT.IfChange(forked_exports)
"""Helper methods for legacy features"""

load(":common/cc/action_names.bzl", "ACTION_NAMES")
load(
    ":common/cc/toolchain_config/cc_toolchain_config_lib.bzl",
    "action_config",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "tool",
    "variable_with_value",
    "with_feature_set",
)

def get_legacy_features(platform, existing_feature_names, linker_tool_path):
    """The features added to all legacy toolchains

    Note: these features won't be added to the crosstools that defines
    no_legacy_features feature (e.g. ndk, apple, enclave crosstools). Those need
    to be modified separately.

    Args:
        platform: (str) One of 'linux' or 'mac'
        existing_feature_names: ([str])
        linker_tool_path: (str)

    Returns:
        ([FeatureInfo])
    """
    result = []
    if "legacy_compile_flags" not in existing_feature_names:
        result.append(feature(
            name = "legacy_compile_flags",
            enabled = True,
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.clif_match,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [
                    flag_group(
                        expand_if_available = "legacy_compile_flags",
                        iterate_over = "legacy_compile_flags",
                        flags = ["%{legacy_compile_flags}"],
                    ),
                ],
            )],
        ))

    # Gcc options:
    #  -MD turns on .d file output as a side-effect (doesn't imply -E)
    #  -MM[D] enables user includes only, not system includes
    #  -MF <name> specifies the dotd file name
    # Issues:
    #  -M[M] alone subverts actual .o output (implies -E)
    #  -M[M]D alone breaks some of the .d naming assumptions
    # This combination gets user and system includes with specified name:
    #  -MD -MF <name>
    if "dependency_file" not in existing_feature_names:
        result.append(feature(
            name = "dependency_file",
            enabled = True,
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.clif_match,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "dependency_file",
                    flags = ["-MD", "-MF", "%{dependency_file}"],
                )],
            )],
        ))

    # GCC and Clang give randomized names to symbols which are defined in
    # an anonymous namespace but have external linkage.  To make
    # computation of these deterministic, we want to override the
    # default seed for the random number generator.  It's safe to use
    # any value which differs for all translation units; we use the
    # path to the object file.
    if "random_seed" not in existing_feature_names:
        result.append(feature(
            name = "random_seed",
            enabled = True,
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "output_file",
                    flags = ["-frandom-seed=%{output_file}"],
                )],
            )],
        ))

    if "pic" not in existing_feature_names:
        result.append(feature(
            name = "pic",
            enabled = True,
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "pic",
                    flags = ["-fPIC"],
                )],
            )],
        ))

    if "per_object_debug_info" not in existing_feature_names:
        result.append(feature(
            name = "per_object_debug_info",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "per_object_debug_info_file",
                    flags = ["-gsplit-dwarf", "-g"],
                )],
            )],
        ))

    if "preprocessor_defines" not in existing_feature_names:
        result.append(feature(
            name = "preprocessor_defines",
            enabled = True,
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.clif_match,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [flag_group(
                    iterate_over = "preprocessor_defines",
                    flags = ["-D%{preprocessor_defines}"],
                )],
            )],
        ))

    if "includes" not in existing_feature_names:
        result.append(feature(
            name = "includes",
            enabled = True,
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.clif_match,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "includes",
                    iterate_over = "includes",
                    flags = ["-include", "%{includes}"],
                )],
            )],
        ))

    if "include_paths" not in existing_feature_names:
        result.append(feature(
            name = "include_paths",
            enabled = True,
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.clif_match,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [
                    flag_group(
                        iterate_over = "quote_include_paths",
                        flags = ["-iquote", "%{quote_include_paths}"],
                    ),
                    flag_group(
                        iterate_over = "include_paths",
                        flags = ["-I%{include_paths}"],
                    ),
                    flag_group(
                        iterate_over = "system_include_paths",
                        flags = ["-isystem", "%{system_include_paths}"],
                    ),
                    flag_group(
                        iterate_over = "framework_include_paths",
                        flags = ["-F%{framework_include_paths}"],
                    ),
                ],
            )],
        ))

    if "fdo_instrument" not in existing_feature_names:
        result.append(feature(
            name = "fdo_instrument",
            provides = ["profile"],
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_executable,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        expand_if_available = "fdo_instrument_path",
                        flags = ["-fprofile-generate=%{fdo_instrument_path}", "-fno-data-sections"],
                    ),
                ],
            )],
        ))

    if "fdo_optimize" not in existing_feature_names:
        result.append(feature(
            name = "fdo_optimize",
            provides = ["profile"],
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "fdo_profile_path",
                    flags = [
                        "-fprofile-use=%{fdo_profile_path}",
                        "-Wno-profile-instr-unprofiled",
                        "-Wno-profile-instr-out-of-date",
                        "-fprofile-correction",
                    ],
                )],
            )],
        ))

    if "cs_fdo_instrument" not in existing_feature_names:
        result.append(feature(
            name = "cs_fdo_instrument",
            provides = ["csprofile"],
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_executable,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "cs_fdo_instrument_path",
                    flags = ["-fcs-profile-generate=%{cs_fdo_instrument_path}"],
                )],
            )],
        ))

    if "cs_fdo_optimize" not in existing_feature_names:
        result.append(feature(
            name = "cs_fdo_optimize",
            provides = ["csprofile"],
            flag_sets = [flag_set(
                actions = [ACTION_NAMES.lto_backend],
                flag_groups = [flag_group(
                    expand_if_available = "fdo_profile_path",
                    flags = [
                        "-fprofile-use=%{fdo_profile_path}",
                        "-Wno-profile-instr-unprofiled",
                        "-Wno-profile-instr-out-of-date",
                        "-fprofile-correction",
                    ],
                )],
            )],
        ))

    if "fdo_prefetch_hints" not in existing_feature_names:
        result.append(feature(
            name = "fdo_prefetch_hints",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.lto_backend,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "fdo_prefetch_hints_path",
                    flags = [
                        "-mllvm",
                        "-prefetch-hints-file=%{fdo_prefetch_hints_path}",
                    ],
                )],
            )],
        ))

    if "autofdo" not in existing_feature_names:
        result.append(feature(
            name = "autofdo",
            provides = ["profile"],
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "fdo_profile_path",
                    flags = [
                        "-fauto-profile=%{fdo_profile_path}",
                        "-fprofile-correction",
                    ],
                )],
            )],
        ))

    if "propeller_optimize_thinlto_compile_actions" not in existing_feature_names:
        result.append(feature(
            name = "propeller_optimize_thinlto_compile_actions",
        ))

    if "propeller_optimize" not in existing_feature_names:
        result.append(feature(
            name = "propeller_optimize",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.lto_backend,
                    ],
                    flag_groups = [flag_group(
                        expand_if_available = "propeller_optimize_cc_path",
                        flags = [
                            "-fbasic-block-sections=list=%{propeller_optimize_cc_path}",
                            "-DBUILD_PROPELLER_ENABLED=1",
                        ],
                    )],
                ),
                flag_set(
                    actions = [ACTION_NAMES.cpp_link_executable],
                    flag_groups = [flag_group(
                        expand_if_true = "propeller_optimize_ld_path",
                        flags = ["-Wl,--symbol-ordering-file=%{propeller_optimize_ld_path}"],
                    )],
                ),
            ],
        ))

    if "memprof_optimize" not in existing_feature_names:
        result.append(feature(
            name = "memprof_optimize",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "memprof_profile_path",
                    flags = ["-memprof-profile-file=%{memprof_profile_path}"],
                )],
            )],
        ))

    if "build_interface_libraries" not in existing_feature_names:
        result.append(feature(
            name = "build_interface_libraries",
            flag_sets = [flag_set(
                with_features = [with_feature_set(
                    features = ["supports_interface_shared_libraries"],
                )],
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "generate_interface_library",
                    flags = [
                        "%{generate_interface_library}",
                        "%{interface_library_builder_path}",
                        "%{interface_library_input_path}",
                        "%{interface_library_output_path}",
                    ],
                )],
            )],
        ))

    # Order of feature declaration matters, linker_tool_path has to
    # follow right after build_interface_libraries.
    if "dynamic_library_linker_tool" not in existing_feature_names:
        result.append(feature(
            name = "dynamic_library_linker_tool",
            flag_sets = [flag_set(
                with_features = [with_feature_set(
                    features = ["supports_interface_shared_libraries"],
                )],
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "generate_interface_library",
                    flags = [linker_tool_path],
                )],
            )],
        ))

    if "shared_flag" not in existing_feature_names:
        result.append(feature(
            name = "shared_flag",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(flags = ["-shared"])],
            )],
        ))

    if "linkstamps" not in existing_feature_names:
        result.append(feature(
            name = "linkstamps",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_executable,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "linkstamp_paths",
                    iterate_over = "linkstamp_paths",
                    flags = ["%{linkstamp_paths}"],
                )],
            )],
        ))

    if "output_execpath_flags" not in existing_feature_names:
        result.append(feature(
            name = "output_execpath_flags",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_executable,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "output_execpath",
                    flags = ["-o", "%{output_execpath}"],
                )],
            )],
        ))

    if "runtime_library_search_directories" not in existing_feature_names:
        result.append(feature(
            name = "runtime_library_search_directories",
            flag_sets = [
                flag_set(
                    with_features = [with_feature_set(
                        features = ["static_link_cpp_runtimes"],
                    )],
                    actions = [
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.lto_index_for_dynamic_library,
                        ACTION_NAMES.lto_index_for_executable,
                        ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                    ],
                    flag_groups = [
                        flag_group(
                            expand_if_available = "runtime_library_search_directories",
                            iterate_over = "runtime_library_search_directories",
                            flag_groups = [
                                flag_group(
                                    expand_if_true = "is_cc_test",
                                    flags = [
                                        # TODO(b/27153401): This should probably be @loader_path on osx.
                                        "-Xlinker",
                                        "-rpath",
                                        "-Xlinker",
                                        "$EXEC_ORIGIN/%{runtime_library_search_directories}",
                                    ],
                                ),
                                flag_group(
                                    expand_if_false = "is_cc_test",
                                    flags = [
                                        "-Xlinker",
                                        "-rpath",
                                        "-Xlinker",
                                        _platform_specific_value(
                                            platform,
                                            linux = "$ORIGIN/%{runtime_library_search_directories}",
                                            mac = "@loader_path/%{runtime_library_search_directories}",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                flag_set(
                    with_features = [with_feature_set(
                        not_features = ["static_link_cpp_runtimes"],
                    )],
                    actions = [
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.lto_index_for_dynamic_library,
                        ACTION_NAMES.lto_index_for_executable,
                        ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                    ],
                    flag_groups = [flag_group(
                        expand_if_available = "runtime_library_search_directories",
                        iterate_over = "runtime_library_search_directories",
                        flag_groups = [flag_group(
                            flags = [
                                "-Xlinker",
                                "-rpath",
                                "-Xlinker",
                                _platform_specific_value(
                                    platform,
                                    linux = "$ORIGIN/%{runtime_library_search_directories}",
                                    mac = "@loader_path/%{runtime_library_search_directories}",
                                ),
                            ],
                        )],
                    )],
                ),
            ],
        ))

    if "library_search_directories" not in existing_feature_names:
        result.append(feature(
            name = "library_search_directories",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_executable,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "library_search_directories",
                    iterate_over = "library_search_directories",
                    flags = ["-L%{library_search_directories}"],
                )],
            )],
        ))

    if "archiver_flags" not in existing_feature_names:
        result.append(feature(
            name = "archiver_flags",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.cpp_link_static_library],
                    flag_groups = [
                        flag_group(
                            flags = [_platform_specific_value(
                                platform,
                                linux = "rcsD",
                                mac = "-static",
                            )],
                        ),
                        flag_group(
                            expand_if_available = "output_execpath",
                            flags = _platform_specific_value(
                                platform,
                                linux = [],
                                mac = ["-o"],
                            ) + [
                                "%{output_execpath}",
                            ],
                        ),
                    ],
                ),
                flag_set(
                    actions = [ACTION_NAMES.cpp_link_static_library],
                    flag_groups = [
                        flag_group(
                            expand_if_available = "libraries_to_link",
                            iterate_over = "libraries_to_link",
                            flag_groups = [
                                flag_group(
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "object_file",
                                    ),
                                    flags = ["%{libraries_to_link.name}"],
                                ),
                                flag_group(
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "object_file_group",
                                    ),
                                    iterate_over = "libraries_to_link.object_files",
                                    flags = ["%{libraries_to_link.object_files}"],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ))

    if "libraries_to_link" not in existing_feature_names:
        result.append(feature(
            name = "libraries_to_link",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_executable,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        expand_if_true = "thinlto_param_file",
                        flags = ["-Wl,@%{thinlto_param_file}"],
                    ),
                    flag_group(
                        expand_if_available = "libraries_to_link",
                        iterate_over = "libraries_to_link",
                        flag_groups = [
                            flag_group(
                                expand_if_equal = variable_with_value(
                                    "libraries_to_link.type",
                                    "object_file_group",
                                ),
                                expand_if_false = "libraries_to_link.is_whole_archive",
                                flags = ["-Wl,--start-lib"],
                            ),
                        ] + _platform_specific_value(
                            platform,
                            linux = [
                                flag_group(
                                    expand_if_true = "libraries_to_link.is_whole_archive",
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "static_library",
                                    ),
                                    flags = ["-Wl,-whole-archive"],
                                ),
                                flag_group(
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "object_file_group",
                                    ),
                                    iterate_over = "libraries_to_link.object_files",
                                    flags = ["%{libraries_to_link.object_files}"],
                                ),
                                flag_group(
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "object_file",
                                    ),
                                    flags = ["%{libraries_to_link.name}"],
                                ),
                                flag_group(
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "interface_library",
                                    ),
                                    flags = ["%{libraries_to_link.name}"],
                                ),
                                flag_group(
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "static_library",
                                    ),
                                    flags = ["%{libraries_to_link.name}"],
                                ),
                                flag_group(
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "dynamic_library",
                                    ),
                                    flags = ["-l%{libraries_to_link.name}"],
                                ),
                                flag_group(
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "versioned_dynamic_library",
                                    ),
                                    flags = ["-l:%{libraries_to_link.name}"],
                                ),
                                flag_group(
                                    expand_if_true = "libraries_to_link.is_whole_archive",
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "static_library",
                                    ),
                                    flags = ["-Wl,-no-whole-archive"],
                                ),
                            ],
                            # macOS mirrors the linux behavior above with the following exceptions:
                            # - The -Wl,-whole-archive flags are excluded, as they are not supported on
                            #   macOS.
                            # - Every expansion of libraries_to_link is split into a -Wl,-force_load
                            #   version and a regular link version (with the exception of
                            #   dynamic_library and versioned_dynamic_library).
                            # - versioned_dynamic_library has slightly different flag syntax on
                            #   macOS.
                            mac = [
                                flag_group(
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "object_file_group",
                                    ),
                                    iterate_over = "libraries_to_link.object_files",
                                    flag_groups = [
                                        flag_group(
                                            expand_if_false = "libraries_to_link.is_whole_archive",
                                            flags = ["%{libraries_to_link.object_files}"],
                                        ),
                                        flag_group(
                                            expand_if_true = "libraries_to_link.is_whole_archive",
                                            flags = ["-Wl,-force_load,%{libraries_to_link.object_files}"],
                                        ),
                                    ],
                                ),
                                flag_group(
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "object_file",
                                    ),
                                    flag_groups = [
                                        flag_group(
                                            expand_if_false = "libraries_to_link.is_whole_archive",
                                            flags = ["%{libraries_to_link.name}"],
                                        ),
                                        flag_group(
                                            expand_if_true = "libraries_to_link.is_whole_archive",
                                            flags = ["-Wl,-force_load,%{libraries_to_link.name}"],
                                        ),
                                    ],
                                ),
                                flag_group(
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "interface_library",
                                    ),
                                    flag_groups = [
                                        flag_group(
                                            expand_if_false = "libraries_to_link.is_whole_archive",
                                            flags = ["%{libraries_to_link.name}"],
                                        ),
                                        flag_group(
                                            expand_if_true = "libraries_to_link.is_whole_archive",
                                            flags = ["-Wl,-force_load,%{libraries_to_link.name}"],
                                        ),
                                    ],
                                ),
                                flag_group(
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "static_library",
                                    ),
                                    flag_groups = [
                                        flag_group(
                                            expand_if_false = "libraries_to_link.is_whole_archive",
                                            flags = ["%{libraries_to_link.name}"],
                                        ),
                                        flag_group(
                                            expand_if_true = "libraries_to_link.is_whole_archive",
                                            flags = ["-Wl,-force_load,%{libraries_to_link.name}"],
                                        ),
                                    ],
                                ),
                                flag_group(
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "dynamic_library",
                                    ),
                                    flags = ["-l%{libraries_to_link.name}"],
                                ),
                                flag_group(
                                    expand_if_equal = variable_with_value(
                                        "libraries_to_link.type",
                                        "versioned_dynamic_library",
                                    ),
                                    flags = ["%{libraries_to_link.path}"],
                                ),
                            ],
                        ) + [
                            flag_group(
                                expand_if_equal = variable_with_value(
                                    "libraries_to_link.type",
                                    "object_file_group",
                                ),
                                expand_if_false = "libraries_to_link.is_whole_archive",
                                flags = ["-Wl,--end-lib"],
                            ),
                        ],
                    ),
                ],
            )],
        ))

    if "force_pic_flags" not in existing_feature_names:
        result.append(feature(
            name = "force_pic_flags",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.lto_index_for_executable,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "force_pic",
                    flags = [
                        _platform_specific_value(
                            platform,
                            linux = "-pie",
                            mac = "-Wl,-pie",
                        ),
                    ],
                )],
            )],
        ))

    if "user_link_flags" not in existing_feature_names:
        result.append(feature(
            name = "user_link_flags",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_executable,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "user_link_flags",
                    iterate_over = "user_link_flags",
                    flags = ["%{user_link_flags}"],
                )],
            )],
        ))

    if "legacy_link_flags" not in existing_feature_names:
        result.append(feature(
            name = "legacy_link_flags",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_executable,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "legacy_link_flags",
                    iterate_over = "legacy_link_flags",
                    flags = ["%{legacy_link_flags}"],
                )],
            )],
        ))

    if "static_libgcc" not in existing_feature_names:
        result.append(feature(
            name = "static_libgcc",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_executable,
                ],
                with_features = [
                    with_feature_set(features = ["static_link_cpp_runtimes"]),
                ],
                flag_groups = [flag_group(flags = ["-static-libgcc"])],
            )],
        ))

    if "fission_support" not in existing_feature_names:
        result.append(feature(
            name = "fission_support",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_executable,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "is_using_fission",
                    flags = ["-Wl,--gdb-index"],
                )],
            )],
        ))

    if "strip_debug_symbols" not in existing_feature_names:
        result.append(feature(
            name = "strip_debug_symbols",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_executable,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "strip_debug_symbols",
                    flags = ["-Wl,-S"],
                )],
            )],
        ))

    if "coverage" not in existing_feature_names:
        result.extend([
            feature(name = "coverage"),
            feature(
                name = "llvm_coverage_map_format",
                provides = ["profile"],
                flag_sets = [
                    flag_set(
                        actions = [
                            ACTION_NAMES.c_compile,
                            ACTION_NAMES.cpp_compile,
                            ACTION_NAMES.cpp_module_compile,
                            ACTION_NAMES.objc_compile,
                            ACTION_NAMES.objcpp_compile,
                            ACTION_NAMES.preprocess_assemble,
                        ],
                        flag_groups = [flag_group(flags = [
                            "-fprofile-instr-generate",
                            "-fcoverage-mapping",
                        ])],
                    ),
                    flag_set(
                        actions = [
                            ACTION_NAMES.cpp_link_dynamic_library,
                            ACTION_NAMES.cpp_link_executable,
                            ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                            ACTION_NAMES.lto_index_for_dynamic_library,
                            ACTION_NAMES.lto_index_for_executable,
                            ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                            ACTION_NAMES.objc_executable,
                            ACTION_NAMES.objcpp_executable,
                        ],
                        flag_groups = [flag_group(flags = ["-fprofile-instr-generate"])],
                    ),
                ],
                requires = [feature_set(features = ["coverage"])],
            ),
            feature(
                name = "gcc_coverage_map_format",
                provides = ["profile"],
                flag_sets = [
                    flag_set(
                        actions = [
                            ACTION_NAMES.c_compile,
                            ACTION_NAMES.cpp_compile,
                            ACTION_NAMES.cpp_module_compile,
                            ACTION_NAMES.objc_compile,
                            ACTION_NAMES.objc_executable,
                            ACTION_NAMES.objcpp_compile,
                            ACTION_NAMES.objcpp_executable,
                            ACTION_NAMES.preprocess_assemble,
                        ],
                        flag_groups = [flag_group(
                            expand_if_available = "gcov_gcno_file",
                            flags = ["-fprofile-arcs", "-ftest-coverage"],
                        )],
                    ),
                    flag_set(
                        actions = [
                            ACTION_NAMES.cpp_link_dynamic_library,
                            ACTION_NAMES.cpp_link_executable,
                            ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                            ACTION_NAMES.lto_index_for_dynamic_library,
                            ACTION_NAMES.lto_index_for_executable,
                            ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                        ],
                        flag_groups = [flag_group(flags = ["--coverage"])],
                    ),
                ],
                requires = [feature_set(features = ["coverage"])],
            ),
        ])

    return result

def get_features_to_appear_last(existing_feature_names):
    """Set of legacy features to append at the end of the feature list

    Note:  these feaures won't be added to the crosstools that defines
    no_legacy_features feature (e.g. ndk, apple, enclave crosstools). Those need
    to be modified separately.

    Args:
        existing_feature_names: (str)

    Returns:
        ([FeatureInfo])
    """
    result = []

    if "fully_static_link" not in existing_feature_names:
        result.append(feature(
            name = "fully_static_link",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_executable,
                ],
                flag_groups = [flag_group(flags = ["-static"])],
            )],
        ))

    if "user_compile_flags" not in existing_feature_names:
        result.append(feature(
            name = "user_compile_flags",
            enabled = True,
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.clif_match,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "user_compile_flags",
                    iterate_over = "user_compile_flags",
                    flags = ["%{user_compile_flags}"],
                )],
            )],
        ))

    if "sysroot" not in existing_feature_names:
        result.append(feature(
            name = "sysroot",
            enabled = True,
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.clif_match,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_executable,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "sysroot",
                    flags = ["--sysroot=%{sysroot}"],
                )],
            )],
        ))

    # unfiltered_compile_flags contain system include paths. These must be added
    # after the user provided options (present in legacy_compile_flags build
    # variable above), otherwise users adding include paths will not pick up their
    # own include paths first.
    if "unfiltered_compile_flags" not in existing_feature_names:
        result.append(feature(
            name = "unfiltered_compile_flags",
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.clif_match,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "unfiltered_compile_flags",
                    iterate_over = "unfiltered_compile_flags",
                    flags = ["%{unfiltered_compile_flags}"],
                )],
            )],
        ))

    if "linker_param_file" not in existing_feature_names:
        result.append(feature(
            name = "linker_param_file",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.lto_index_for_dynamic_library,
                        ACTION_NAMES.lto_index_for_executable,
                        ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                    ],
                    flag_groups = [flag_group(
                        expand_if_available = "linker_param_file",
                        flags = ["@%{linker_param_file}"],
                    )],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.cpp_link_static_library,
                    ],
                    flag_groups = [flag_group(
                        expand_if_available = "linker_param_file",
                        flags = ["@%{linker_param_file}"],
                    )],
                ),
            ],
        ))

    if "compiler_input_flags" not in existing_feature_names:
        result.append(feature(
            name = "compiler_input_flags",
            enabled = True,
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [flag_group(
                    expand_if_available = "source_file",
                    flags = ["-c", "%{source_file}"],
                )],
            )],
        ))

    if "compiler_output_flags" not in existing_feature_names:
        result.append(feature(
            name = "compiler_output_flags",
            enabled = True,
            flag_sets = [flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [
                    flag_group(
                        expand_if_available = "output_assembly_file",
                        flags = ["-S"],
                    ),
                    flag_group(
                        expand_if_available = "output_preprocess_file",
                        flags = ["-E"],
                    ),
                    flag_group(
                        expand_if_available = "output_file",
                        flags = ["-o", "%{output_file}"],
                    ),
                ],
            )],
        ))

    return result

def get_legacy_action_configs(
        platform,
        gcc_tool_path,
        ar_tool_path,
        strip_tool_path,
        existing_action_config_names):
    """The list of action configs added to all legacy toolchains

    Note:  these configs won't be added to the crosstools that defines
    no_legacy_features feature (e.g. ndk, apple, enclave crosstools). Those need
    to be modified separately.

    Args:
        platform: (str) 'mac' or 'linux'
        gcc_tool_path: (str)
        ar_tool_path: (str)
        strip_tool_path: (str)
        existing_action_config_names: ([str])

    Returns:
        ([ActionConfigInfo])
    """
    result = []

    for action_name in [
        "assemble",
        "preprocess-assemble",
        "linkstamp-compile",
        "lto-backend",
        "c-compile",
        "c++-compile",
        "c++-header-parsing",
        "c++-module-compile",
        "c++-module-codegen",
    ]:
        if action_name not in existing_action_config_names:
            result.append(action_config(
                action_name = action_name,
                tools = [tool(path = gcc_tool_path)],
                implies = [
                    "legacy_compile_flags",
                    "user_compile_flags",
                    "sysroot",
                    "unfiltered_compile_flags",
                    "compiler_input_flags",
                    "compiler_output_flags",
                ],
            ))

    for action_name in [
        "c++-link-executable",
        "lto-index-for-executable",
    ]:
        if action_name not in existing_action_config_names:
            result.append(action_config(
                action_name = action_name,
                tools = [tool(path = gcc_tool_path)],
                implies = [
                    "strip_debug_symbols",
                    "linkstamps",
                    "output_execpath_flags",
                    "runtime_library_search_directories",
                    "library_search_directories",
                    "libraries_to_link",
                    "force_pic_flags",
                    "user_link_flags",
                    "legacy_link_flags",
                    "linker_param_file",
                    "fission_support",
                    "sysroot",
                ],
            ))

    for action_name in [
        "c++-link-nodeps-dynamic-library",
        "lto-index-for-nodeps-dynamic-library",
        "c++-link-dynamic-library",
        "lto-index-for-dynamic-library",
    ]:
        if action_name not in existing_action_config_names:
            result.append(action_config(
                action_name = action_name,
                tools = [tool(path = gcc_tool_path)],
                implies = [
                    "build_interface_libraries",
                    "dynamic_library_linker_tool",
                    "strip_debug_symbols",
                    "shared_flag",
                    "linkstamps",
                    "output_execpath_flags",
                    "runtime_library_search_directories",
                    "library_search_directories",
                    "libraries_to_link",
                    "user_link_flags",
                    "legacy_link_flags",
                    "linker_param_file",
                    "fission_support",
                    "sysroot",
                ],
            ))

    if "c++-link-static-library" not in existing_action_config_names:
        result.append(action_config(
            action_name = "c++-link-static-library",
            tools = [tool(path = ar_tool_path)],
            implies = ["archiver_flags", "linker_param_file"],
        ))

    if "strip" not in existing_action_config_names:
        result.append(action_config(
            action_name = "strip",
            tools = [tool(path = strip_tool_path)],
            flag_sets = [flag_set(
                flag_groups = [
                    flag_group(
                        flags = ["-S"] + _platform_specific_value(
                            platform,
                            linux = ["-p"],
                            mac = [],
                        ) + ["-o", "%{output_file}"],
                    ),
                    flag_group(
                        iterate_over = "stripopts",
                        flags = ["%{stripopts}"],
                    ),
                    flag_group(flags = ["%{input_file}"]),
                ],
            )],
        ))

    return result

def _platform_specific_value(platform, *, linux, mac):
    if platform == "linux":
        return linux
    if platform == "mac":
        return mac
    fail("unexpected platform:", platform)

# LINT.ThenChange(@rules_cc//cc/private/toolchain_config/legacy_features.bzl:forked_exports)
