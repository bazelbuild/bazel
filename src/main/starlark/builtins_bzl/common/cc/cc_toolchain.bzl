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

load(":common/cc/cc_toolchain_provider_helper.bzl", "get_cc_toolchain_provider")
load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/objc/objc_common.bzl", "objc_common")

cc_internal = _builtins.internal.cc_internal
ToolchainInfo = _builtins.toplevel.platform_common.ToolchainInfo
TemplateVariableInfo = _builtins.toplevel.platform_common.TemplateVariableInfo
apple_common = _builtins.toplevel.apple_common

def _validate_toolchain(ctx, is_apple):
    if not is_apple:
        return
    if ctx.attr._xcode_config[apple_common.XcodeVersionConfig].xcode_version() == None:
        fail("Xcode version must be specified to use an Apple CROSSTOOL. If your Xcode version has " +
             "changed recently, verify that \"xcode-select -p\" is correct and then try: " +
             "\"bazel shutdown\" to re-run Xcode configuration")

def _cc_toolchain_impl(ctx):
    _validate_toolchain(ctx, ctx.attr._is_apple)
    if ctx.attr._is_apple:
        build_vars_func = objc_common.apple_cc_toolchain_build_variables(ctx.attr._xcode_config[apple_common.XcodeVersionConfig])
    else:
        build_vars_func = cc_helper.cc_toolchain_build_variables(None)

    attributes_provider = cc_internal.construct_cc_toolchain_attributes_info(
        ctx = ctx,
        is_apple = ctx.attr._is_apple,
        build_vars_func = build_vars_func,
    )
    providers = [attributes_provider]
    if attributes_provider.licenses_provider() != None:
        providers.append(attributes_provider.licenses_provider())
    if not ctx.fragments.cpp.enable_cc_toolchain_resolution():
        # This is not a platforms-backed build, let's provide CcToolchainAttributesProvider
        # and have cc_toolchain_suite select one of its toolchains and create CcToolchainProvider
        # from its attributes. We also need to provide a do-nothing ToolchainInfo.
        providers.append(ToolchainInfo(cc = "dummy cc toolchain"))
        return providers
    cc_toolchain = get_cc_toolchain_provider(ctx, attributes_provider, ctx.attr._is_apple)
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
