# pylint: disable=g-bad-file-header
# Copyright 2018 The Bazel Authors. All rights reserved.
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
"""Library of common crosstool features."""

load(
    "@bazel_tools//tools/cpp:crosstool_utils.bzl",
    "ARCHIVE_ACTIONS",
    "COMPILE_ACTIONS",
    "LINK_ACTIONS",
    "feature",
    "flag_group",
    "flag_set",
    "flags",
    "simple_feature",
)

def get_features_to_appear_first(platform):
    """Returns standard features that should appear in the top of the toolchain.

    Args:
      platform: one of [ k8, darwin, msvc ]

    Returns:
      a collection of features to be put into crosstool
    """
    return [
        simple_feature("no_legacy_features", [], []),
        simple_feature(
            "legacy_compile_flags",
            COMPILE_ACTIONS,
            ["%{legacy_compile_flags}"],
            expand_if_all_available = ["legacy_compile_flags"],
            iterate_over = "legacy_compile_flags",
        ),
        simple_feature(
            "dependency_file",
            COMPILE_ACTIONS,
            ["-MD", "-MF", "%{dependency_file}"],
            expand_if_all_available = ["dependency_file"],
        ),
        simple_feature(
            "random_seed",
            COMPILE_ACTIONS,
            ["-frandom-seed=%{output_file}"],
        ),
        simple_feature(
            "pic",
            COMPILE_ACTIONS,
            ["-fPIC"],
            expand_if_all_available = ["pic"],
        ),
        simple_feature(
            "per_object_debug_info",
            COMPILE_ACTIONS,
            ["-gsplit-dwarf"],
            expand_if_all_available = ["per_object_debug_info_file"],
        ),
        simple_feature(
            "preprocessor_defines",
            COMPILE_ACTIONS,
            ["-D%{preprocessor_defines}"],
            iterate_over = "preprocessor_defines",
            expand_if_all_available = ["preprocessor_defines"],
        ),
        simple_feature(
            "includes",
            COMPILE_ACTIONS,
            ["-include", "%{includes}"],
            iterate_over = "includes",
            expand_if_all_available = ["includes"],
        ),
        simple_feature(
            "quote_include_paths",
            COMPILE_ACTIONS,
            ["-iquote", "%{quote_include_paths}"],
            iterate_over = "quote_include_paths",
            expand_if_all_available = ["quote_include_paths"],
        ),
        simple_feature(
            "include_paths",
            COMPILE_ACTIONS,
            ["-I%{include_paths}"],
            iterate_over = "include_paths",
            expand_if_all_available = ["include_paths"],
        ),
        simple_feature(
            "system_include_paths",
            COMPILE_ACTIONS,
            ["-isystem", "%{system_include_paths}"],
            iterate_over = "system_include_paths",
            expand_if_all_available = ["system_include_paths"],
        ),
        simple_feature(
            "symbol_counts",
            LINK_ACTIONS,
            ["-Wl,--print-symbol-counts=%{symbol_counts_output}"],
            expand_if_all_available = ["symbol_counts_output"],
        ),
        simple_feature(
            "shared_flag",
            LINK_ACTIONS,
            ["-shared"],
            expand_if_all_available = ["symbol_counts_output"],
        ),
        simple_feature(
            "output_execpath_flags",
            LINK_ACTIONS,
            ["-o", "%{output_execpath}"],
            expand_if_all_available = ["output_execpath"],
        ),
        simple_feature(
            "runtime_library_search_directories",
            LINK_ACTIONS,
            [_runtime_library_directory_flag(platform)],
            iterate_over = "runtime_library_search_directories",
            expand_if_all_available = ["runtime_library_search_directories"],
        ),
        simple_feature(
            "library_search_directories",
            LINK_ACTIONS,
            ["-L%{library_search_directories}"],
            iterate_over = "library_search_directories",
            expand_if_all_available = ["library_search_directories"],
        ),
        simple_feature("_archiver_flags", ARCHIVE_ACTIONS, _archiver_flags(platform)),
        feature(
            "libraries_to_link",
            [
                flag_set(ARCHIVE_ACTIONS, [
                    flag_group(
                        [
                            flag_group(
                                flags("%{libraries_to_link.name}"),
                                expand_if_equal = [["libraries_to_link.type", "object_file"]],
                            ),
                            flag_group(
                                flags("%{libraries_to_link.object_files}"),
                                expand_if_equal = [["libraries_to_link.type", "object_file_group"]],
                                iterate_over = "libraries_to_link.object_files",
                            ),
                        ],
                        iterate_over = "libraries_to_link",
                        expand_if_all_available = ["libraries_to_link"],
                    ),
                ]),
                flag_set(LINK_ACTIONS, [
                    flag_group(
                        [
                            flag_group(
                                flags("-Wl,--start-lib"),
                                expand_if_equal = [["libraries_to_link.type", "object_file_group"]],
                            ),
                        ] +
                        _libraries_to_link_flag_groupss(platform) + [
                            flag_group(
                                flags("-Wl,--end-lib"),
                                expand_if_equal = [["libraries_to_link.type", "object_file_group"]],
                            ),
                        ],
                        iterate_over = "libraries_to_link",
                    ),
                    flag_group(flags("-Wl,@%{thinlto_param_file}"), expand_if_true = ["thinlto_param_file"]),
                ]),
            ],
        ),
        simple_feature(
            "force_pic_flags",
            ["c++-link-executable"],
            [_force_pic_flag(platform)],
            expand_if_all_available = ["force_pic"],
        ),
        simple_feature(
            "user_link_flags",
            LINK_ACTIONS,
            ["%{user_link_flags}"],
            iterate_over = "user_link_flags",
            expand_if_all_available = ["user_link_flags"],
        ),
        simple_feature(
            "legacy_link_flags",
            LINK_ACTIONS,
            ["%{legacy_link_flags}"],
            iterate_over = "legacy_link_flags",
            expand_if_all_available = ["legacy_link_flags"],
        ),
        simple_feature(
            "fission_support",
            LINK_ACTIONS,
            ["-Wl,--gdb-index"],
            expand_if_all_available = ["is_using_fission"],
        ),
        simple_feature(
            "strip_debug_symbols",
            LINK_ACTIONS,
            ["-Wl,-S"],
            expand_if_all_available = ["strip_debug_symbols"],
        ),
        _coverage_feature(platform),
        simple_feature("strip_flags", ["strip"], _strip_flags(platform)),
    ]

def get_features_to_appear_last(platform):
    """Returns standard features that should appear at the end of the toolchain.

    Args:
      platform: one of [ k8, darwin, msvc ]

    Returns:
      a collection of features to be put into crosstool
    """
    return [
        simple_feature(
            "user_compile_flags",
            COMPILE_ACTIONS,
            ["%{user_compile_flags}"],
            expand_if_all_available = ["user_compile_flags"],
            iterate_over = "user_compile_flags",
        ),
        simple_feature(
            "sysroot",
            COMPILE_ACTIONS + LINK_ACTIONS,
            ["--sysroot=%{sysroot}"],
            expand_if_all_available = ["sysroot"],
        ),
        simple_feature(
            "unfiltered_compile_flags",
            COMPILE_ACTIONS,
            ["%{unfiltered_compile_flags}"],
            expand_if_all_available = ["unfiltered_compile_flags"],
            iterate_over = "unfiltered_compile_flags",
        ),
        simple_feature(
            "linker_param_file",
            LINK_ACTIONS,
            [_linker_param_file_flag(platform)],
            expand_if_all_available = ["linker_param_file"],
        ),
        simple_feature(
            "archiver_param_file",
            ARCHIVE_ACTIONS,
            [_archiver_param_file_flag(platform)],
            expand_if_all_available = ["linker_param_file"],
        ),
        simple_feature(
            "compiler_input_flags",
            COMPILE_ACTIONS,
            ["-c", "%{source_file}"],
            expand_if_all_available = ["source_file"],
        ),
        feature(
            "compiler_output_flags",
            [
                flag_set(COMPILE_ACTIONS, [
                    flag_group(
                        flags("-S"),
                        expand_if_all_available = ["output_assembly_file"],
                    ),
                    flag_group(
                        flags("-E"),
                        expand_if_all_available = ["output_preprocess_file"],
                    ),
                    flag_group(
                        flags("-o", "%{output_file}"),
                        expand_if_all_available = ["output_file"],
                    ),
                ]),
            ],
        ),
    ]

def _is_linux(platform):
    return platform == "k8"

def _is_darwin(platform):
    return platform == "darwin"

def _is_msvc(platform):
    return platform == "msvc"

def _coverage_feature(use_llvm_format):
    if use_llvm_format:
        compile_flags = flags("-fprofile-instr-generate", "-fcoverage-mapping")
        link_flags = flags("-fprofile-instr-generate")
    else:
        compile_flags = flags("-fprofile-arcs", "-ftest-coverage")
        link_flags = flags("--coverage")
    return feature(
        "coverage",
        [
            flag_set(COMPILE_ACTIONS, [flag_group(compile_flags)]),
            flag_set(LINK_ACTIONS, [flag_group(link_flags)]),
        ],
        enabled = False,
        provides = "profile",
    )

def _runtime_library_directory_flag(platform):
    if _is_linux(platform):
        return "-Wl,-rpath,$ORIGIN/%{runtime_library_search_directories}"
    elif _is_darwin(platform):
        return "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}"
    elif _is_msvc(platform):
        fail("todo")
    else:
        fail("Unsupported platform: " + platform)

def _force_pic_flag(platform):
    if _is_linux(platform):
        return "-pie"
    elif _is_darwin(platform):
        return "-Wl,-pie"
    elif _is_msvc(platform):
        fail("todo")
    else:
        fail("Unsupported platform: " + platform)

def _archiver_flags(platform):
    if _is_linux(platform):
        return ["rcsD", "%{output_execpath}"]
    elif _is_darwin(platform):
        return ["-static", "-s", "-o", "%{output_execpath}"]
    elif _is_msvc(platform):
        fail("todo")
    else:
        fail("Unsupported platform: " + platform)

def _library_to_link_with_worce_load(variable_type, variable, flag = "", iterate = False):
    return [
        flag_group(
            [
                flag_group(
                    flags(
                        "-Wl,-force_load," + flag + "%{" + variable + "}",
                        expand_if_true = ["libraries_to_link.is_whole_archive"],
                    ),
                ),
                flag_group(
                    flags(
                        flag + "%{" + variable + "}",
                        expand_if_false = ["libraries_to_link.is_whole_archive"],
                    ),
                ),
            ],
            iterate_over = variable if iterate else None,
            expand_if_equal = [["libraries_to_link.type", variable_type]],
        ),
    ]

def _libraries_to_link_flag_groupss(platform):
    if _is_linux(platform):
        return [
            flag_group(
                flags("-Wl,-whole-archive"),
                expand_if_true = ["libraries_to_link.is_whole_archive"],
            ),
            flag_group(
                flags("-Wl,--start-lib"),
                expand_if_equal = [["libraries_to_link.type", "object_file_group"]],
            ),
            flag_group(
                flags("%{libraries_to_link.object_files}"),
                iterate_over = "libraries_to_link.object_files",
                expand_if_equal = [["libraries_to_link.type", "object_file_group"]],
            ),
            flag_group(
                flags("-Wl,--end-lib"),
                expand_if_equal = [["libraries_to_link.type", "object_file_group"]],
            ),
            flag_group(
                flags("%{libraries_to_link.name}"),
                expand_if_equal = [["libraries_to_link.type", "object_file"]],
            ),
            flag_group(
                flags("%{libraries_to_link.name}"),
                expand_if_equal = [["libraries_to_link.type", "interface_library"]],
            ),
            flag_group(
                flags("%{libraries_to_link.name}"),
                expand_if_equal = [["libraries_to_link.type", "static_library"]],
            ),
            flag_group(
                flags("-l%{libraries_to_link.name}"),
                expand_if_equal = [["libraries_to_link.type", "dynamic_library"]],
            ),
            flag_group(
                flags("-l:%{libraries_to_link.name}"),
                expand_if_equal = [["libraries_to_link.type", "versioned_dynamic_library"]],
            ),
            flag_group(
                flags("-Wl,-no-whole-archive"),
                expand_if_true = ["libraries_to_link.is_whole_archive"],
            ),
        ]
    if _is_darwin(platform):
        return [
            flag_group(
                flags("-Wl,--start-lib"),
                expand_if_equal = [["libraries_to_link.type", "object_file_group"]],
            ),
            _library_to_link_with_worce_load(
                "object_file_group",
                "libraries_to_link.object_files",
                iterate = True,
            ),
            flag_group(
                flags("-Wl,--end-lib"),
                expand_if_equal = [["libraries_to_link.type", "object_file_group"]],
            ),
            _library_to_link_with_worce_load("object_file", "libraries_to_link.name"),
            _library_to_link_with_worce_load("interface_library", "libraries_to_link.name"),
            _library_to_link_with_worce_load("static_library", "libraries_to_link.name"),
            _library_to_link_with_worce_load("dynamic_library", "libraries_to_link.name", flag = "-l"),
            _library_to_link_with_worce_load("versioned_dynamic_library", "libraries_to_link.name", flag = "-l:"),
        ]
    elif _is_msvc(platform):
        fail("todo")
    else:
        fail("Unsupported platform: " + platform)

def _strip_flags(platform):
    if _is_linux(platform):
        return [
            "-S",
            "-p",
            "-o",
            "%{output_file}",
            "-R",
            ".gnu.switches.text.quote_paths",
            "-R",
            ".gnu.switches.text.bracket_paths",
            "-R",
            ".gnu.switches.text.system_paths",
            "-R",
            ".gnu.switches.text.cpp_defines",
            "-R",
            ".gnu.switches.text.cpp_includes",
            "-R",
            ".gnu.switches.text.cl_args",
            "-R",
            ".gnu.switches.text.lipo_info",
            "-R",
            ".gnu.switches.text.annotation",
        ]
    elif _is_darwin(platform):
        return ["-S", "-o", "%{output_file}"]
    elif _is_msvc(platform):
        fail("todo")
    else:
        fail("Unsupported platform: " + platform)

def _linker_param_file_flag(platform):
    if _is_linux(platform):
        return "-Wl,@%{linker_param_file}"
    elif _is_darwin(platform):
        return "-Wl,@%{linker_param_file}"
    elif _is_msvc(platform):
        fail("todo")
    else:
        fail("Unsupported platform: " + platform)

def _archiver_param_file_flag(platform):
    if _is_linux(platform):
        return "@%{linker_param_file}"
    elif _is_darwin(platform):
        return "@%{linker_param_file}"
    elif _is_msvc(platform):
        fail("todo")
    else:
        fail("Unsupported platform: " + platform)
