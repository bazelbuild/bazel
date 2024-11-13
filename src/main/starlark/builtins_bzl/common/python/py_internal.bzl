# Copyright 2022 The Bazel Authors. All rights reserved.
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
"""PYTHON RULE IMPLEMENTATION ONLY: Do not use outside of the rule implementations and their tests.

Various builtin Starlark defined objects exposed for non-builtin Starlark.

These may change at any time and are closely coupled to the rule implementation.
"""

load(":common/cc/cc_common.bzl", "cc_common")
load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/cc/semantics.bzl", cc_semantics = "semantics")

_py_builtins = _builtins.internal.py_builtins
PackageSpecificationInfo = _builtins.toplevel.PackageSpecificationInfo

# NOTE: Directly exporting the Java objects on _py_builtins results in some
# stricter serialization check failures within the Google implementation.
# Instead, they must be wrapped in a Starlark function.

def _add_py_extra_pseudo_action(*args, **kwargs):
    return _py_builtins.add_py_extra_pseudo_action(*args, **kwargs)

def _are_action_listeners_enabled(*args, **kwargs):
    return _py_builtins.are_action_listeners_enabled(*args, **kwargs)

def _cc_info_debug_context(cc_info, *args, **kwargs):
    return cc_info.debug_context(*args, **kwargs)

def _cc_semantics_get_cc_runtimes(*args, **kwargs):
    return cc_semantics.get_cc_runtimes(*args, **kwargs)

def _cc_semantics_get_runtimes_toolchain(*args, **kwargs):
    return cc_semantics.get_runtimes_toolchain(*args, **kwargs)

def _cc_semantics_get_stl(*args, **kwargs):
    return cc_semantics.get_stl(*args, **kwargs)

def _cc_toolchain_strip_files(cc_toolchain, *args, **kwargs):
    return cc_toolchain._strip_files

def _cc_toolchain_build_info_files(cc_toolchain, *args, **kwargs):
    return cc_toolchain._build_info_files

def _cc_launcher_info_cc_info(cc_launcher_info, *args, **kwargs):
    return cc_launcher_info.cc_info(*args, **kwargs)

def _compilation_outputs(cc_launcher_info):
    return cc_launcher_info.compilation_outputs()

def _compilation_outputs_gcno_files(compilation_outputs, *args, **kwargs):
    return compilation_outputs.gcno_files(*args, **kwargs)

def _compilation_outputs_pic_gcno_files(compilation_outputs, *args, **kwargs):
    return compilation_outputs.pic_gcno_files(*args, **kwargs)

def _compile(*args, **kwargs):
    return cc_common.compile(*args, **kwargs)

def _copy_without_caching(*args, **kwargs):
    return _py_builtins.copy_without_caching(*args, **kwargs)

def _create_linking_context_from_compilation_outputs(*args, **kwargs):
    return cc_common.create_linking_context_from_compilation_outputs(*args, **kwargs)

def _create_repo_mapping_manifest(*args, **kwargs):
    return _py_builtins.create_repo_mapping_manifest(*args, **kwargs)

def _create_sources_only_manifest(*args, **kwargs):
    return _py_builtins.create_sources_only_manifest(*args, **kwargs)

def _declare_constant_metadata_file(*args, **kwargs):
    return _py_builtins.declare_constant_metadata_file(*args, **kwargs)

def _declare_shareable_artifact(ctx, *args, **kwargs):
    return ctx.actions.declare_shareable_artifact(*args, **kwargs)

def _expand_location_and_make_variables(*args, **kwargs):
    return _py_builtins.expand_location_and_make_variables(*args, **kwargs)

def _extra_libraries_build_libraries(extra_libraries, *args, **kwargs):
    return extra_libraries.build_libraries(*args, **kwargs)

def _get_current_os_name(*args, **kwargs):
    return _py_builtins.get_current_os_name(*args, **kwargs)

def _get_label_repo_runfiles_path(*args, **kwargs):
    return _py_builtins.get_label_repo_runfiles_path(*args, **kwargs)

def _get_legacy_external_runfiles(*args, **kwargs):
    return _py_builtins.get_legacy_external_runfiles(*args, **kwargs)

def _get_rule_name(*args, **kwargs):
    return _py_builtins.get_rule_name(*args, **kwargs)

def _is_available_for(package_group_target, label):
    return package_group_target[PackageSpecificationInfo].contains(label)

def _is_bzlmod_enabled(*args, **kwargs):
    return _py_builtins.is_bzlmod_enabled(*args, **kwargs)

def _is_singleton_depset(*args, **kwargs):
    return _py_builtins.is_singleton_depset(*args, **kwargs)

def _is_tool_configuration(ctx):
    return ctx.configuration.is_tool_configuration()

def _regex_match(*args, **kwargs):
    return _py_builtins.regex_match(*args, **kwargs)

def _runfiles_enabled(ctx):
    return ctx.configuration.runfiles_enabled()

def _link(*args, **kwargs):
    return cc_common.link(*args, **kwargs)

def _linking_context_extra_link_time_libraries(linking_context, *args, **kwargs):
    return linking_context.extra_link_time_libraries(*args, **kwargs)

def _linking_context_linkstamps(linking_context, *args, **kwargs):
    return linking_context.linkstamps(*args, **kwargs)

def _linkstamp_file(linkstamp, *args, **kwargs):
    return linkstamp.file(*args, **kwargs)

def _make_runfiles_respect_legacy_external_runfiles(*args, **kwargs):
    return _py_builtins.make_runfiles_respect_legacy_external_runfiles(*args, **kwargs)

def _merge_debug_context(*args, **kwargs):
    return cc_common.merge_debug_context(*args, **kwargs)

def _merge_linking_contexts(*args, **kwargs):
    return cc_common.merge_linking_contexts(*args, **kwargs)

def _merge_runfiles_with_generated_inits_empty_files_supplier(*args, **kwargs):
    return _py_builtins.merge_runfiles_with_generated_inits_empty_files_supplier(*args, **kwargs)

def _share_native_deps(ctx):
    return ctx.fragments.cpp.share_native_deps()

def _stamp_binaries(ctx):
    return ctx.configuration.stamp_binaries()

def _strip_opts(ctx):
    return ctx.fragments.cpp.strip_opts()

# This replaces the Java-defined name using exports.bzl toplevels mapping.
py_internal = struct(
    CcLauncherInfo = _builtins.internal.cc_internal.launcher_provider,
    PackageSpecificationInfo = PackageSpecificationInfo,
    add_py_extra_pseudo_action = _add_py_extra_pseudo_action,
    are_action_listeners_enabled = _are_action_listeners_enabled,
    cc_helper = cc_helper,
    cc_info_debug_context = _cc_info_debug_context,
    cc_launcher_info_cc_info = _cc_launcher_info_cc_info,
    cc_semantics_get_cc_runtimes = _cc_semantics_get_cc_runtimes,
    cc_semantics_get_runtimes_toolchain = _cc_semantics_get_runtimes_toolchain,
    cc_semantics_get_stl = _cc_semantics_get_stl,
    cc_toolchain_strip_files = _cc_toolchain_strip_files,
    cc_toolchain_build_info_files = _cc_toolchain_build_info_files,
    compilation_outputs = _compilation_outputs,
    compilation_outputs_gcno_files = _compilation_outputs_gcno_files,
    compilation_outputs_pic_gcno_files = _compilation_outputs_pic_gcno_files,
    compile = _compile,
    copy_without_caching = _copy_without_caching,
    create_linking_context_from_compilation_outputs = _create_linking_context_from_compilation_outputs,
    create_repo_mapping_manifest = _create_repo_mapping_manifest,
    create_sources_only_manifest = _create_sources_only_manifest,
    declare_constant_metadata_file = _declare_constant_metadata_file,
    declare_shareable_artifact = _declare_shareable_artifact,
    expand_location_and_make_variables = _expand_location_and_make_variables,
    extra_libraries_build_libraries = _extra_libraries_build_libraries,
    get_current_os_name = _get_current_os_name,
    get_label_repo_runfiles_path = _get_label_repo_runfiles_path,
    get_legacy_external_runfiles = _get_legacy_external_runfiles,
    get_rule_name = _get_rule_name,
    is_available_for = _is_available_for,
    is_bzlmod_enabled = _is_bzlmod_enabled,
    is_singleton_depset = _is_singleton_depset,
    is_tool_configuration = _is_tool_configuration,
    link = _link,
    linking_context_extra_link_time_libraries = _linking_context_extra_link_time_libraries,
    linking_context_linkstamps = _linking_context_linkstamps,
    linkstamp_file = _linkstamp_file,
    make_runfiles_respect_legacy_external_runfiles = _make_runfiles_respect_legacy_external_runfiles,
    merge_debug_context = _merge_debug_context,
    merge_linking_contexts = _merge_linking_contexts,
    merge_runfiles_with_generated_inits_empty_files_supplier = _merge_runfiles_with_generated_inits_empty_files_supplier,
    regex_match = _regex_match,
    runfiles_enabled = _runfiles_enabled,
    share_native_deps = _share_native_deps,
    stamp_binaries = _stamp_binaries,
    strip_opts = _strip_opts,
)
