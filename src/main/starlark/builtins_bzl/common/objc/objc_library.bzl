# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""objc_library Starlark implementation replacing native"""

load("@_builtins//:common/objc/compilation_support.bzl", "compilation_support")
load("@_builtins//:common/objc/attrs.bzl", "common_attrs")
load("@_builtins//:common/objc/objc_common.bzl", "extensions")
load("@_builtins//:common/objc/transitions.bzl", "apple_crosstool_transition")
load("@_builtins//:common/cc/cc_helper.bzl", "cc_helper")

objc_internal = _builtins.internal.objc_internal
CcInfo = _builtins.toplevel.CcInfo
cc_common = _builtins.toplevel.cc_common
coverage_common = _builtins.toplevel.coverage_common
apple_common = _builtins.toplevel.apple_common

def _attribute_error(attr_name, msg):
    fail("in attribute '" + attr_name + "': " + msg)

def _validate_attributes(label):
    if label.name.find("/") != -1:
        _attribute_error("name", "this attribute has unsupported character '/'")

def _objc_library_impl(ctx):
    """Implementation of objc_library."""

    _validate_attributes(label = ctx.label)

    cc_toolchain = cc_helper.find_cpp_toolchain(ctx)

    common_variables = compilation_support.build_common_variables(
        ctx = ctx,
        toolchain = cc_toolchain,
        use_pch = True,
        deps = ctx.attr.deps,
        runtime_deps = ctx.attr.runtime_deps,
        attr_linkopts = ctx.attr.linkopts,
        alwayslink = ctx.attr.alwayslink,
    )
    files = []
    if common_variables.compilation_artifacts.archive != None:
        files.append(common_variables.compilation_artifacts.archive)

    compilation_result = compilation_support.register_compile_and_archive_actions(
        common_variables,
    )
    (compilation_context, linking_context, compilation_outputs, output_groups) = compilation_result

    compilation_support.validate_attributes(common_variables)

    j2objc_providers = objc_internal.j2objc_providers_from_deps(ctx = ctx)

    objc_provider = common_variables.objc_provider

    instrumented_files_info = coverage_common.instrumented_files_info(
        ctx = ctx,
        source_attributes = ["srcs", "non_arc_srcs", "hdrs"],
        dependency_attributes = ["deps", "data", "binary", "xctest_app"],
        extensions = extensions.NON_CPP_SOURCES + extensions.CPP_SOURCES + extensions.HEADERS,
        coverage_environment = cc_helper.get_coverage_environment(ctx, ctx.fragments.cpp, cc_toolchain),
        coverage_support_files = cc_toolchain.coverage_files() if ctx.coverage_instrumented() else depset([]),
        metadata_files = compilation_outputs.gcno_files() + compilation_outputs.pic_gcno_files(),
    )

    return [
        DefaultInfo(
            files = depset(files),
            data_runfiles = ctx.runfiles(files = files),
        ),
        CcInfo(
            compilation_context = compilation_context,
            linking_context = linking_context,
        ),
        objc_provider,
        j2objc_providers[0],
        j2objc_providers[1],
        instrumented_files_info,
        OutputGroupInfo(**output_groups),
    ]

objc_library = rule(
    implementation = _objc_library_impl,
    attrs = common_attrs.union(
        {
            "data": attr.label_list(allow_files = True),
        },
        common_attrs.CC_TOOLCHAIN_RULE,
        common_attrs.LICENSES,
        common_attrs.COMPILING_RULE,
        common_attrs.COMPILE_DEPENDENCY_RULE,
        common_attrs.INCLUDE_SCANNING_RULE,
        common_attrs.SDK_FRAMEWORK_DEPENDER_RULE,
        common_attrs.COPTS_RULE,
        common_attrs.XCRUN_RULE,
    ),
    fragments = ["objc", "apple", "cpp"],
    cfg = apple_crosstool_transition,
    toolchains = cc_helper.use_cpp_toolchain(),
    incompatible_use_toolchain_transition = True,
)
