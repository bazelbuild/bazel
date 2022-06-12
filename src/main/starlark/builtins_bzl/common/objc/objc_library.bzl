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

load("@_builtins//:common/objc/semantics.bzl", "semantics")
load("@_builtins//:common/objc/compilation_support.bzl", "compilation_support")
load("@_builtins//:common/objc/attrs.bzl", "common_attrs")
load("@_builtins//:common/objc/transitions.bzl", "apple_crosstool_transition")
load("@_builtins//:common/cc/cc_helper.bzl", "cc_helper")

objc_internal = _builtins.internal.objc_internal
CcInfo = _builtins.toplevel.CcInfo
cc_common = _builtins.toplevel.cc_common
coverage_common = _builtins.toplevel.coverage_common
apple_common = _builtins.toplevel.apple_common

def _attribute_error(attr_name, msg):
    fail("in attribute '" + attr_name + "': " + msg)

def _validate_attributes(ctx):
    if ctx.label.name.find("/") != -1:
        _attribute_error("name", "this attribute has unsupported character '/'")

def _build_linking_context(ctx, feature_configuration, cc_toolchain, objc_provider, common_variables):
    libraries = []
    if common_variables.compilation_artifacts.archive != None:
        library_to_link = _static_library(ctx, feature_configuration, cc_toolchain, common_variables.compilation_artifacts.archive)
        libraries.append(library_to_link)

    archives_from_objc_library = {}
    for library in objc_provider.library.to_list():
        archives_from_objc_library[library.path] = library

    objc_libraries_cc_infos = []
    for dep in ctx.attr.deps:
        if apple_common.Objc in dep and CcInfo in dep:
            objc_libraries_cc_infos.append(dep[CcInfo])

    merged_objc_library_cc_infos = cc_common.merge_cc_infos(cc_infos = objc_libraries_cc_infos)

    for linker_input in merged_objc_library_cc_infos.linking_context.linker_inputs.to_list():
        for lib in linker_input.libraries:
            path = None
            if lib.static_library != None:
                path = lib.static_library.path
            elif lib.pic_static_library != None:
                path = lib.pic_static_library.path
            if path in archives_from_objc_library and archives_from_objc_library[path]:
                libraries.append(lib)
                archives_from_objc_library[path] = None

    for archive in archives_from_objc_library.values():
        if archive:
            library_to_link = _static_library(ctx, feature_configuration, cc_toolchain, archive)
            libraries.append(library_to_link)

    libraries.extend(objc_provider.cc_library.to_list())

    user_link_flags = _user_link_flags(
        cc_info = merged_objc_library_cc_infos,
        objc_provider = objc_provider,
    )

    direct_linker_inputs = []
    if len(user_link_flags) != 0 or len(libraries) != 0 or objc_provider.linkstamp:
        linker_input = cc_common.create_linker_input(
            owner = ctx.label,
            libraries = depset(libraries),
            user_link_flags = user_link_flags,
            linkstamps = objc_provider.linkstamp,
        )
        direct_linker_inputs.append(linker_input)

    return cc_common.create_linking_context(
        linker_inputs = depset(direct = direct_linker_inputs, order = "topological"),
    )

def _static_library(
        ctx,
        feature_configuration,
        cc_toolchain,
        library):
    alwayslink = False
    if library.extension == "lo":
        alwayslink = True
    return cc_common.create_library_to_link(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        static_library = library,
        alwayslink = alwayslink,
    )

def _user_link_flags(*, cc_info, objc_provider):
    """Builds objc_library CcInfo user link flags for frameworks and dylibs.

    Args:
        cc_info: Merged CcInfo provider from objc_library target deps.
        objc_provider: Current objc_library ObjC provider.
    Returns:
        List of user link flags for frameworks and dylibs.
    """

    sdk_dylibs = objc_provider.sdk_dylib.to_list()
    sdk_frameworks = objc_provider.sdk_framework.to_list()

    all_user_link_flags = []
    all_user_link_flags.extend(objc_provider.linkopt.to_list())

    for linker_input in cc_info.linking_context.linker_inputs.to_list():
        all_user_link_flags.extend(linker_input.user_link_flags)

    for i, user_link_flag in enumerate(all_user_link_flags):
        if user_link_flag.startswith("-l"):
            sdk_dylibs.append("lib" + user_link_flag[2:])
        elif user_link_flag == "-framework":
            sdk_frameworks.append(all_user_link_flags[i + 1])

    sdk_user_link_flags = []
    for sdk_framework in depset(sdk_frameworks).to_list():
        sdk_user_link_flags.append(["-framework", sdk_framework])
    for sdk_dylib in depset(sdk_dylibs).to_list():
        if sdk_dylib.startswith("lib"):
            sdk_dylib = sdk_dylib[3:]
        sdk_user_link_flags.append(["-l" + sdk_dylib])

    return sdk_user_link_flags

def _objc_library_impl(ctx):
    _validate_attributes(ctx)

    cc_toolchain = cc_helper.find_cpp_toolchain(ctx)

    common_variables = compilation_support.build_common_variables(
        ctx = ctx,
        toolchain = cc_toolchain,
        use_pch = True,
        deps = ctx.attr.deps,
        runtime_deps = ctx.attr.runtime_deps,
        linkopts = ctx.attr.linkopts,
        alwayslink = ctx.attr.alwayslink,
    )
    files = []
    if common_variables.compilation_artifacts.archive != None:
        files.append(common_variables.compilation_artifacts.archive)

    (cc_compilation_context, compilation_outputs, output_groups) = compilation_support.register_compile_and_archive_actions(
        common_variables,
    )

    compilation_support.validate_attributes(common_variables)

    j2objc_providers = objc_internal.j2objc_providers_from_deps(ctx = ctx)

    objc_provider = common_variables.objc_provider
    feature_configuration = compilation_support.build_feature_configuration(common_variables, False, True)
    linking_context = _build_linking_context(ctx, feature_configuration, cc_toolchain, objc_provider, common_variables)
    cc_info = CcInfo(
        compilation_context = cc_compilation_context,
        linking_context = linking_context,
    )

    return [
        DefaultInfo(files = depset(files), data_runfiles = ctx.runfiles(files = files)),
        cc_info,
        objc_provider,
        j2objc_providers[0],
        j2objc_providers[1],
        objc_internal.instrumented_files_info(
            ctx = ctx,
            cc_toolchain = cc_toolchain,
            config = ctx.configuration,
            object_files = compilation_outputs.objects,
        ),
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
    toolchains = ["@" + semantics.get_repo() + "//tools/cpp:toolchain_type"],
    incompatible_use_toolchain_transition = True,
)
