# Copyright 2025 The Bazel Authors. All rights reserved.
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
"""Helper functions for C++ feature configuration."""

load(":common/cc/action_names.bzl", "ACTION_NAMES")
load(":common/cc/semantics.bzl", cc_semantics = "semantics")

_cc_common_internal = _builtins.internal.cc_common

ALL_COMPILE_ACTIONS = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.cpp_module_deps_scanning,
    ACTION_NAMES.cpp20_module_compile,
    ACTION_NAMES.cpp20_module_codegen,
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.clif_match,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.cc_flags_make_variable,
    ACTION_NAMES.lto_backend,
    ACTION_NAMES.cpp_header_analysis,
]

ALL_LINK_ACTIONS = [
    ACTION_NAMES.lto_index_for_executable,
    ACTION_NAMES.lto_index_for_dynamic_library,
    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

ALL_ARCHIVE_ACTIONS = [
    ACTION_NAMES.cpp_link_static_library,
]

ALL_OTHER_ACTIONS = [
    ACTION_NAMES.strip,
]

DEFAULT_ACTION_CONFIGS = ALL_COMPILE_ACTIONS + ALL_LINK_ACTIONS + ALL_ARCHIVE_ACTIONS + ALL_OTHER_ACTIONS

OBJC_ACTIONS = [
    ACTION_NAMES.objc_compile,
    ACTION_NAMES.objcpp_compile,
    ACTION_NAMES.objc_fully_link,
    ACTION_NAMES.objc_executable,
]

def _get_coverage_features(cpp_configuration):
    coverage_features = []
    coverage_features.append("coverage")
    if cpp_configuration.use_llvm_coverage_map_format():
        coverage_features.append("llvm_coverage_map_format")
    else:
        coverage_features.append("gcc_coverage_map_format")
    return coverage_features

def configure_features(
        *,
        ctx,
        cc_toolchain,
        language = "c++",
        requested_features = [],
        unsupported_features = []):
    """Creates a feature_configuration instance. Requires the cpp configuration fragment.

    Args:
      ctx: (RuleContext) The rule context.
      cc_toolchain: (CcToolchainInfo) cc_toolchain for which we configure features.
      language: ("c++"|"objc"|""objc++") The language to configure for. (default c++)
      requested_features: (list[str]) List of features to be enabled.
      unsupported_features: (list[str]) List of features that are unsupported by the current rule.

    Returns:
      (FeatureConfiguration) The feature configuration.
    """

    language = (language or "c++").replace("+", "p")

    # TODO(b/236152224): Remove the following when all Starlark objc configure_features have the
    # chance to migrate to using the language parameter.
    if "lang_objc" in requested_features:
        language = "objc"

    if not hasattr(ctx.fragments, "cpp"):
        fail("cpp configuration fragment is missing")

    cpp_configuration = ctx.fragments.cpp

    if language == "cpp":
        cc_semantics.validate_layering_check_features(
            ctx = ctx,
            cc_toolchain = cc_toolchain,
            unsupported_features = unsupported_features,
        )

    all_requested_features_set = set()
    all_unsupported_features_set = set(unsupported_features)

    if not cc_toolchain._supports_header_parsing:
        # TODO(b/159096411): Remove once supports_header_parsing has been removed from the
        # cc_toolchain rule.
        all_unsupported_features_set.add("parse_headers")

    if (language != "objc" and
        language != "objcpp" and
        cc_toolchain._cc_info.compilation_context._module_map == None):
        all_unsupported_features_set.add("module_maps")

    if cpp_configuration.force_pic():
        if "supports_pic" in all_unsupported_features_set:
            fail("PIC compilation is requested but the toolchain does not support it " +
                 "(feature named 'supports_pic' is not enabled)")
        all_requested_features_set.add("supports_pic")

    if cpp_configuration.apple_generate_dsym:
        all_requested_features_set.add("generate_dsym_file")
    else:
        all_requested_features_set.add("no_generate_debug_symbols")

    if language == "objc" or language == "objcpp":
        all_requested_features_set.add("lang_objc")
        if cpp_configuration.objc_generate_linkmap:
            all_requested_features_set.add("generate_linkmap")
        if cpp_configuration.objc_should_strip_binary:
            all_requested_features_set.add("dead_strip")

    all_features = [cpp_configuration.compilation_mode()]
    all_features.extend(DEFAULT_ACTION_CONFIGS)
    all_features.extend(requested_features)
    all_features.extend(cc_toolchain._toolchain_features.default_features_and_action_configs())

    if language == "objc" or language == "objcpp":
        all_features.extend(OBJC_ACTIONS)

    if not cpp_configuration._dont_enable_host_nonhost:
        if cc_toolchain._is_tool_configuration:
            all_features.append("host")
        else:
            all_features.append("nonhost")

    if ctx.configuration.coverage_enabled:
        all_features.extend(_get_coverage_features(cpp_configuration))

    if "fdo_instrument" not in all_unsupported_features_set:
        if cpp_configuration.fdo_instrument() != None:
            all_features.append("fdo_instrument")
        elif cpp_configuration.cs_fdo_instrument() != None:
            all_features.append("cs_fdo_instrument")

    fdo_context = cc_toolchain._fdo_context
    branch_fdo_provider = getattr(fdo_context, "branch_fdo_profile", None)

    enable_propeller_optimize = (
        getattr(fdo_context, "propeller_optimize_info", None) != None and
        (fdo_context.propeller_optimize_info.cc_profile != None or
         fdo_context.propeller_optimize_info.ld_profile != None)
    )

    if branch_fdo_provider != None and cpp_configuration.compilation_mode() == "opt":
        if ((branch_fdo_provider.branch_fdo_mode == "llvm_fdo" or
             branch_fdo_provider.branch_fdo_mode == "llvm_cs_fdo") and
            "fdo_optimize" not in all_unsupported_features_set):
            all_features.append("fdo_optimize")
            if "thin_lto" not in all_unsupported_features_set:
                all_features.append("enable_fdo_thinlto")
            if ("split_functions" not in all_unsupported_features_set and
                not enable_propeller_optimize):
                all_features.append("enable_fdo_split_functions")

        if branch_fdo_provider.branch_fdo_mode == "llvm_cs_fdo":
            all_features.append("cs_fdo_optimize")

        if branch_fdo_provider.branch_fdo_mode == "auto_fdo":
            all_features.append("autofdo")
            if "memprof_optimize" not in all_unsupported_features_set:
                all_features.append("enable_autofdo_memprof_optimize")
            if "thin_lto" not in all_unsupported_features_set:
                all_features.append("enable_afdo_thinlto")
            if "fsafdo" not in all_unsupported_features_set:
                all_features.append("enable_fsafdo")
                if "split_functions" not in all_unsupported_features_set:
                    all_features.append("enable_fdo_split_functions")

        if branch_fdo_provider.branch_fdo_mode == "xbinary_fdo":
            all_features.append("xbinaryfdo")
            if "thin_lto" not in all_unsupported_features_set:
                all_features.append("enable_xbinaryfdo_thinlto")

    if cpp_configuration._fdo_prefetch_hints_label != None:
        all_requested_features_set.add("fdo_prefetch_hints")

    if enable_propeller_optimize:
        all_requested_features_set.add("propeller_optimize")

    for feature in all_features:
        if feature not in all_unsupported_features_set:
            all_requested_features_set.add(feature)

    feature_configuration = cc_toolchain._toolchain_features.configure_features(
        requested_features = list(all_requested_features_set),
    )

    for feature in all_unsupported_features_set:
        if feature_configuration.is_enabled(feature):
            fail(("The C++ toolchain '{}' unconditionally implies feature '{}', which is unsupported " +
                  "by this rule. This is most likely a misconfiguration in the C++ toolchain.")
                .format(cc_toolchain._toolchain_label, feature))

    if (cpp_configuration.force_pic() and
        not feature_configuration.is_enabled("pic") and
        not feature_configuration.is_enabled("supports_pic")):
        fail("PIC compilation is requested but the toolchain does not support it " +
             "(feature named 'supports_pic' is not enabled)")

    return feature_configuration
