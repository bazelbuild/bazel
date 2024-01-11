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

"""Starlark implementation of cc_toolchain rule."""

load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/cc/cc_toolchain_provider_helper.bzl", "get_cc_toolchain_provider")
load(":common/cc/semantics.bzl", "semantics")

cc_internal = _builtins.internal.cc_internal
ToolchainInfo = _builtins.toplevel.platform_common.ToolchainInfo
TemplateVariableInfo = _builtins.toplevel.platform_common.TemplateVariableInfo
apple_common = _builtins.toplevel.apple_common
FdoProfileInfo = _builtins.internal.FdoProfileInfo
FdoPrefetchHintsInfo = _builtins.internal.FdoPrefetchHintsInfo
PropellerOptimizeInfo = _builtins.internal.PropellerOptimizeInfo
PackageSpecificationInfo = _builtins.toplevel.PackageSpecificationInfo
CcToolchainConfigInfo = _builtins.toplevel.CcToolchainConfigInfo
MemProfProfileInfo = _builtins.internal.MemProfProfileInfo

def _validate_toolchain(ctx, is_apple):
    if not is_apple:
        return
    if ctx.attr._xcode_config[apple_common.XcodeVersionConfig].xcode_version() == None:
        fail("Xcode version must be specified to use an Apple CROSSTOOL. If your Xcode version has " +
             "changed recently, verify that \"xcode-select -p\" is correct and then try: " +
             "\"bazel shutdown\" to re-run Xcode configuration")

def _files(ctx, attr_name):
    attr = getattr(ctx.attr, attr_name, None)
    if attr != None and DefaultInfo in attr:
        return attr[DefaultInfo].files
    return depset()

def _provider(attr, provider):
    if attr != None and provider in attr:
        return attr[provider]
    return None

def _latebound_libc(ctx, attr_name, implicit_attr_name):
    if getattr(ctx.attr, implicit_attr_name, None) == None:
        return attr_name
    return implicit_attr_name

def _full_inputs_for_link(ctx, linker_files, libc, is_apple_toolchain):
    if not is_apple_toolchain:
        return depset(
            [ctx.file._interface_library_builder, ctx.file._link_dynamic_library_tool],
            transitive = [linker_files, libc],
        )
    return depset(transitive = [linker_files, libc])

def _label(ctx, attr_name):
    if getattr(ctx.attr, attr_name, None) != None:
        return getattr(ctx.attr, attr_name).label
    return None

def _package_specification_provider(ctx, allowlist_name):
    possible_attr_names = ["_whitelist_" + allowlist_name, "_allowlist_" + allowlist_name]
    for attr_name in possible_attr_names:
        if hasattr(ctx.attr, attr_name):
            package_specification_provider = getattr(ctx.attr, attr_name)[PackageSpecificationInfo]
            if package_specification_provider != None:
                return package_specification_provider
    fail("Allowlist argument for " + allowlist_name + " not found")

def _single_file(ctx, attr_name):
    files = getattr(ctx.files, attr_name, [])
    if len(files) > 1:
        fail(ctx.label.name + " expected a single artifact", attr = attr_name)
    if len(files) == 1:
        return files[0]
    return None

def _attributes(ctx, is_apple):
    grep_includes = None
    if not semantics.is_bazel:
        grep_includes = _single_file(ctx, "_grep_includes")

    latebound_libc = _latebound_libc(ctx, "libc_top", "_libc_top")
    latebound_target_libc = _latebound_libc(ctx, "libc_top", "_target_libc_top")

    all_files = _files(ctx, "all_files")
    return struct(
        supports_param_files = ctx.attr.supports_param_files,
        runtime_solib_dir_base = "_solib__" + cc_internal.escape_label(label = ctx.label),
        fdo_prefetch_provider = _provider(ctx.attr._fdo_prefetch_hints, FdoPrefetchHintsInfo),
        propeller_optimize_provider = _provider(ctx.attr._propeller_optimize, PropellerOptimizeInfo),
        mem_prof_profile_provider = _provider(ctx.attr._memprof_profile, MemProfProfileInfo),
        cc_toolchain_config_info = _provider(ctx.attr.toolchain_config, CcToolchainConfigInfo),
        fdo_optimize_artifacts = ctx.files._fdo_optimize,
        licenses_provider = cc_internal.licenses(ctx = ctx),
        static_runtime_lib = ctx.attr.static_runtime_lib,
        dynamic_runtime_lib = ctx.attr.dynamic_runtime_lib,
        supports_header_parsing = ctx.attr.supports_header_parsing,
        all_files = all_files,
        compiler_files = _files(ctx, "compiler_files"),
        strip_files = _files(ctx, "strip_files"),
        objcopy_files = _files(ctx, "objcopy_files"),
        fdo_optimize_label = _label(ctx, "_fdo_optimize"),
        link_dynamic_library_tool = ctx.file._link_dynamic_library_tool,
        grep_includes = grep_includes,
        module_map = ctx.attr.module_map,
        as_files = _files(ctx, "as_files"),
        ar_files = _files(ctx, "ar_files"),
        dwp_files = _files(ctx, "dwp_files"),
        fdo_optimize_provider = _provider(ctx.attr._fdo_optimize, FdoProfileInfo),
        module_map_artifact = _single_file(ctx, "module_map"),
        all_files_including_libc = depset(transitive = [_files(ctx, "all_files"), _files(ctx, latebound_libc)]),
        fdo_profile_provider = _provider(ctx.attr._fdo_profile, FdoProfileInfo),
        cs_fdo_profile_provider = _provider(ctx.attr._csfdo_profile, FdoProfileInfo),
        x_fdo_profile_provider = _provider(ctx.attr._xfdo_profile, FdoProfileInfo),
        zipper = ctx.file._zipper,
        linker_files = _full_inputs_for_link(
            ctx,
            _files(ctx, "linker_files"),
            _files(ctx, latebound_libc),
            is_apple,
        ),
        cc_toolchain_label = ctx.label,
        coverage_files = _files(ctx, "coverage_files") or all_files,
        compiler_files_without_includes = _files(ctx, "compiler_files_without_includes"),
        libc = _files(ctx, latebound_libc),
        target_libc = _files(ctx, latebound_target_libc),
        libc_top_label = _label(ctx, latebound_libc),
        target_libc_top_label = _label(ctx, latebound_target_libc),
        if_so_builder = ctx.file._interface_library_builder,
        allowlist_for_layering_check = _package_specification_provider(ctx, "disabling_parse_headers_and_layering_check_allowed"),
        build_info_files = _provider(ctx.attr._build_info_translator, OutputGroupInfo),
    )

def _cc_toolchain_impl(ctx):
    _validate_toolchain(ctx, ctx.attr._is_apple)
    xcode_config_info = None
    if ctx.attr._is_apple:
        xcode_config_info = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]
    attributes = _attributes(ctx, ctx.attr._is_apple)
    providers = []
    if attributes.licenses_provider != None:
        providers.append(attributes.licenses_provider)

    cc_toolchain = get_cc_toolchain_provider(ctx, attributes, xcode_config_info)
    if cc_toolchain == None:
        fail("This should never happen")
    template_variable_info = TemplateVariableInfo(
        cc_toolchain.get_additional_make_variables() | cc_helper.get_toolchain_global_make_variables(cc_toolchain),
    )
    toolchain = ToolchainInfo(
        cc = cc_toolchain,
        # Add a clear signal that this is a CcToolchainProvider, since just "cc" is
        # generic enough to possibly be re-used.
        cc_provider_in_toolchain = True,
    )
    providers.append(cc_toolchain)
    providers.append(toolchain)
    providers.append(template_variable_info)
    providers.append(DefaultInfo(files = cc_toolchain.get_all_files_including_libc()))
    return providers

def make_cc_toolchain(cc_toolchain_attrs, **kwargs):
    return rule(
        implementation = _cc_toolchain_impl,
        fragments = ["cpp", "platform", "apple"],
        attrs = cc_toolchain_attrs,
        **kwargs
    )
