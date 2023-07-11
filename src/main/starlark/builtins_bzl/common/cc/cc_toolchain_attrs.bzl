# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""Attributes for cc_toolchain rule."""

load(":common/cc/semantics.bzl", "semantics")

FdoProfileInfo = _builtins.internal.FdoProfileInfo
FdoPrefetchHintsInfo = _builtins.internal.FdoPrefetchHintsInfo
PropellerOptimizeInfo = _builtins.internal.PropellerOptimizeInfo
PackageSpecificationInfo = _builtins.internal.PackageSpecificationInfo
CcToolchainConfigInfo = _builtins.toplevel.CcToolchainConfigInfo
MemProfProfileInfo = _builtins.internal.MemProfProfileInfo

cc_toolchain_attrs_exec = {
    "cpu": attr.string(),
    "compiler": attr.string(),
    # buildifier: disable=attr-license
    "licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
    # buildifier: disable=attr-license
    "output_licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
    "toolchain_identifier": attr.string(default = ""),
    # Start of exec specific attributes.
    "all_files": attr.label(
        allow_files = True,
        mandatory = True,
        cfg = "exec",
    ),
    "compiler_files": attr.label(
        allow_files = True,
        mandatory = True,
        cfg = "exec",
    ),
    "compiler_files_without_includes": attr.label(
        allow_files = True,
        cfg = "exec",
    ),
    "strip_files": attr.label(
        allow_files = True,
        mandatory = True,
        cfg = "exec",
    ),
    "objcopy_files": attr.label(
        allow_files = True,
        mandatory = True,
        cfg = "exec",
    ),
    "as_files": attr.label(
        allow_files = True,
        cfg = "exec",
    ),
    "ar_files": attr.label(
        allow_files = True,
        cfg = "exec",
    ),
    "linker_files": attr.label(
        allow_files = True,
        mandatory = True,
        cfg = "exec",
    ),
    "dwp_files": attr.label(
        allow_files = True,
        mandatory = True,
        cfg = "exec",
    ),
    "coverage_files": attr.label(
        allow_files = True,
        cfg = "exec",
    ),
    # End of exec specific attributes.
    "libc_top": attr.label(
        allow_files = False,
        cfg = "target",
    ),
    "static_runtime_lib": attr.label(
        allow_files = True,
        cfg = "target",
    ),
    "dynamic_runtime_lib": attr.label(
        allow_files = True,
        cfg = "target",
    ),
    "module_map": attr.label(
        allow_files = True,
        cfg = "target",
    ),
    "supports_param_files": attr.bool(
        default = True,
    ),
    "supports_header_parsing": attr.bool(
        default = False,
    ),
    "exec_transition_for_inputs": attr.bool(
        default = True,
    ),
    "toolchain_config": attr.label(
        allow_files = False,
        mandatory = True,
        providers = [CcToolchainConfigInfo],
        cfg = "target",
    ),
    "_libc_top": attr.label(
        default = configuration_field(fragment = "cpp", name = "libc_top"),
        cfg = "target",
    ),
    "_grep_includes": semantics.get_grep_includes(),
    "_interface_library_builder": attr.label(
        default = "@" + semantics.get_repo() + "//tools/cpp:interface_library_builder",
        allow_single_file = True,
        cfg = "exec",
    ),
    "_link_dynamic_library_tool": attr.label(
        default = "@" + semantics.get_repo() + "//tools/cpp:link_dynamic_library",
        allow_single_file = True,
        cfg = "exec",
    ),
    "_cc_toolchain_type": attr.label(default = "@" + semantics.get_repo() + "//tools/cpp:toolchain_type"),
    "_zipper": attr.label(
        default = configuration_field(fragment = "cpp", name = "zipper"),
        allow_single_file = True,
        cfg = "exec",
    ),
    "_default_zipper": attr.label(
        default = configuration_field(fragment = "cpp", name = "default_zipper"),
        allow_single_file = True,
        cfg = "exec",
    ),
    "_target_libc_top": attr.label(
        default = configuration_field(fragment = "cpp", name = "target_libc_top_DO_NOT_USE_ONLY_FOR_CC_TOOLCHAIN"),
        cfg = "target",
    ),
    "_fdo_optimize": attr.label(
        default = configuration_field(fragment = "cpp", name = "fdo_optimize"),
        cfg = "target",
        allow_files = True,
    ),
    "_xfdo_profile": attr.label(
        default = configuration_field(fragment = "cpp", name = "xbinary_fdo"),
        allow_rules = ["fdo_profile"],
        providers = [FdoProfileInfo],
        cfg = "target",
    ),
    "_fdo_profile": attr.label(
        default = configuration_field(fragment = "cpp", name = "fdo_profile"),
        allow_rules = ["fdo_profile"],
        providers = [FdoProfileInfo],
        cfg = "target",
    ),
    "_csfdo_profile": attr.label(
        default = configuration_field(fragment = "cpp", name = "cs_fdo_profile"),
        allow_rules = ["fdo_profile"],
        providers = [FdoProfileInfo],
        cfg = "target",
    ),
    "_fdo_prefetch_hints": attr.label(
        default = configuration_field(fragment = "cpp", name = "fdo_prefetch_hints"),
        allow_rules = ["fdo_prefetch_hints"],
        providers = [FdoPrefetchHintsInfo],
        cfg = "target",
    ),
    "_propeller_optimize": attr.label(
        default = configuration_field(fragment = "cpp", name = "propeller_optimize"),
        allow_rules = ["propeller_optimize"],
        providers = [PropellerOptimizeInfo],
        cfg = "target",
    ),
    "_memprof_profile": attr.label(
        default = configuration_field(fragment = "cpp", name = "memprof_profile"),
        allow_rules = ["memprof_profile"],
        providers = [MemProfProfileInfo],
        cfg = "target",
    ),
    "_whitelist_disabling_parse_headers_and_layering_check_allowed": attr.label(
        default = "@" + semantics.get_repo() + "//tools/build_defs/cc/whitelists/parse_headers_and_layering_check:disabling_parse_headers_and_layering_check_allowed",
        providers = [PackageSpecificationInfo],
        cfg = "exec",
    ),
    "_whitelist_loose_header_check_allowed_in_toolchain": attr.label(
        default = "@" + semantics.get_repo() + "//tools/build_defs/cc/whitelists/starlark_hdrs_check:loose_header_check_allowed_in_toolchain",
        providers = [PackageSpecificationInfo],
        cfg = "exec",
    ),
    "_is_apple": attr.bool(
        default = False,
    ),
}

cc_toolchain_attrs_target = dict(cc_toolchain_attrs_exec)
cc_toolchain_attrs_target["all_files"] = attr.label(
    allow_files = True,
    mandatory = True,
    cfg = "target",
)
cc_toolchain_attrs_target["compiler_files"] = attr.label(
    allow_files = True,
    mandatory = True,
    cfg = "target",
)
cc_toolchain_attrs_target["compiler_files_without_includes"] = attr.label(
    allow_files = True,
    cfg = "target",
)
cc_toolchain_attrs_target["strip_files"] = attr.label(
    allow_files = True,
    mandatory = True,
    cfg = "target",
)
cc_toolchain_attrs_target["objcopy_files"] = attr.label(
    allow_files = True,
    mandatory = True,
    cfg = "target",
)
cc_toolchain_attrs_target["as_files"] = attr.label(
    allow_files = True,
    cfg = "target",
)
cc_toolchain_attrs_target["ar_files"] = attr.label(
    allow_files = True,
    cfg = "target",
)
cc_toolchain_attrs_target["linker_files"] = attr.label(
    allow_files = True,
    mandatory = True,
    cfg = "target",
)
cc_toolchain_attrs_target["dwp_files"] = attr.label(
    allow_files = True,
    mandatory = True,
    cfg = "target",
)
cc_toolchain_attrs_target["coverage_files"] = attr.label(
    allow_files = True,
    cfg = "target",
)

apple_cc_toolchain_attrs_target = dict(cc_toolchain_attrs_target)
apple_cc_toolchain_attrs_exec = dict(cc_toolchain_attrs_exec)

apple_cc_toolchain_attrs_target["_xcode_config"] = attr.label(
    default = configuration_field(fragment = "apple", name = "xcode_config_label"),
    allow_rules = ["xcode_config"],
    flags = ["SKIP_CONSTRAINTS_OVERRIDE"],
)
apple_cc_toolchain_attrs_target["_is_apple"] = attr.bool(
    default = True,
)

apple_cc_toolchain_attrs_exec["_xcode_config"] = attr.label(
    default = configuration_field(fragment = "apple", name = "xcode_config_label"),
    allow_rules = ["xcode_config"],
    flags = ["SKIP_CONSTRAINTS_OVERRIDE"],
)
apple_cc_toolchain_attrs_exec["_is_apple"] = attr.bool(
    default = True,
)
