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
load("@_builtins//:common/cc/cc_helper.bzl", "cc_helper")

objc_internal = _builtins.internal.objc_internal
CcInfo = _builtins.toplevel.CcInfo
cc_common = _builtins.toplevel.cc_common
transition = _builtins.toplevel.transition
coverage_common = _builtins.toplevel.coverage_common
apple_common = _builtins.toplevel.apple_common

def _rule_error(msg):
    fail(msg)

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

    sdk_frameworks = objc_provider.sdk_framework.to_list()
    user_link_flags = []
    for sdk_framework in sdk_frameworks:
        user_link_flags.append(["-framework", sdk_framework])

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

def _to_static_library(
        ctx,
        feature_configuration,
        cc_toolchain,
        library):
    if ((library.pic_static_library == None and
         library.static_library == None) or
        (library.dynamic_library == None and
         library.interface_library == None)):
        return library

    return cc_common.create_library_to_link(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        alwayslink = library.alwayslink,
        pic_objects = library.pic_objects,
        objects = library.objects,
        static_library = library.static_library,
        pic_static_library = library.pic_static_library,
    )

def _objc_library_impl(ctx):
    _validate_attributes(ctx)

    cc_toolchain = cc_helper.find_cpp_toolchain(ctx)

    common_variables = compilation_support.build_common_variables(
        ctx,
        cc_toolchain,
        True,
        False,
        False,
        False,
        ctx.attr.deps,
        ctx.attr.runtime_deps,
        [],
        ctx.attr.linkopts,
    )
    files = []
    if common_variables.compilation_artifacts.archive != None:
        files.append(common_variables.compilation_artifacts.archive)

    (cc_compilation_context, compilation_outputs, output_group_info) = compilation_support.register_compile_and_archive_actions(
        common_variables,
        [],
        [],
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
        objc_internal.instrumented_files_info(ctx = ctx, object_files = compilation_outputs.objects),
        output_group_info,
    ]

def _cpu_string(platform_type, settings):
    arch = _determine_single_architecture(platform_type, settings)
    if platform_type == MACOS:
        return "darwin_{}".format(arch)

    return "{}_{}".format(platform_type, arch)

def _determine_single_architecture(platform_type, settings):
    apple_split_cpu = settings["//command_line_option:apple_split_cpu"]
    if apple_split_cpu != None and len(apple_split_cpu) > 0:
        return apple_split_cpu
    if platform_type == IOS:
        ios_cpus = settings["//command_line_option:ios_multi_cpus"]
        if len(ios_cpus) > 0:
            return ios_cpus[0]
        return _ios_cpu_from_cpu(settings["//command_line_option:cpu"])
    if platform_type == WATCHOS:
        watchos_cpus = settings["//command_line_option:watchos_cpus"]
        if len(watchos_cpus) == 0:
            return DEFAULT_WATCHOS_CPU
        return watchos_cpus[0]
    if platform_type == TVOS:
        tvos_cpus = settings["//command_line_option:tvos_cpus"]
        if len(tvos_cpus) == 0:
            return DEFAULT_TVOS_CPU
        return tvos_cpus[0]
    if platform_type == MACOS:
        macos_cpus = settings["//command_line_option:macos_cpus"]
        if len(macos_cpus) == 0:
            return DEFAULT_MACOS_CPU
        return macos_cpus[0]
    if platform_type == CATALYST:
        catalyst_cpus = settings["//command_line_option:catalyst_cpus"]
        if len(catalyst_cpus) == 0:
            return DEFAULT_CATALYST_CPU
        return catalyst_cpus[0]

    _rule_error("ERROR: Unhandled platform type {}".format(platform_type))
    return None

IOS = "ios"
WATCHOS = "watchos"
TVOS = "tvos"
MACOS = "macos"
CATALYST = "catalyst"
IOS_CPU_PREFIX = "ios_"
DEFAULT_IOS_CPU = "x86_64"
DEFAULT_WATCHOS_CPU = "i386"
DEFAULT_TVOS_CPU = "x86_64"
DEFAULT_MACOS_CPU = "x86_64"
DEFAULT_CATALYST_CPU = "x86_64"

def _ios_cpu_from_cpu(cpu):
    if cpu.startswith(IOS_CPU_PREFIX):
        return cpu[len(IOS_CPU_PREFIX):]
    return DEFAULT_IOS_CPU

def _apple_crosstool_transition_impl(settings, attr):
    platform_type = str(settings["//command_line_option:apple_platform_type"])
    cpu = _cpu_string(platform_type, settings)
    if cpu == settings["//command_line_option:cpu"] and settings["//command_line_option:crosstool_top"] == settings["//command_line_option:apple_crosstool_top"]:
        return {}  # No changes necessary.
    return {
        "//command_line_option:apple configuration distinguisher": "applebin_" + platform_type,
        "//command_line_option:apple_platform_type": settings["//command_line_option:apple_platform_type"],
        "//command_line_option:apple_split_cpu": settings["//command_line_option:apple_split_cpu"],
        "//command_line_option:compiler": settings["//command_line_option:apple_compiler"],
        "//command_line_option:cpu": cpu,
        "//command_line_option:crosstool_top": (
            settings["//command_line_option:apple_crosstool_top"]
        ),
        "//command_line_option:platforms": [],
        "//command_line_option:fission": [],
        "//command_line_option:grte_top": settings["//command_line_option:apple_grte_top"],
        "//command_line_option:ios_minimum_os": settings["//command_line_option:ios_minimum_os"],
        "//command_line_option:macos_minimum_os": settings["//command_line_option:macos_minimum_os"],
        "//command_line_option:tvos_minimum_os": settings["//command_line_option:tvos_minimum_os"],
        "//command_line_option:watchos_minimum_os": settings["//command_line_option:watchos_minimum_os"],
    }

_apple_rule_base_transition_inputs = [
    "//command_line_option:apple configuration distinguisher",
    "//command_line_option:apple_compiler",
    "//command_line_option:compiler",
    "//command_line_option:apple_platform_type",
    "//command_line_option:apple_crosstool_top",
    "//command_line_option:crosstool_top",
    "//command_line_option:apple_split_cpu",
    "//command_line_option:apple_grte_top",
    "//command_line_option:cpu",
    "//command_line_option:ios_multi_cpus",
    "//command_line_option:macos_cpus",
    "//command_line_option:tvos_cpus",
    "//command_line_option:watchos_cpus",
    "//command_line_option:catalyst_cpus",
    "//command_line_option:ios_minimum_os",
    "//command_line_option:macos_minimum_os",
    "//command_line_option:tvos_minimum_os",
    "//command_line_option:watchos_minimum_os",
    "//command_line_option:platforms",
    "//command_line_option:fission",
    "//command_line_option:grte_top",
]
_apple_rule_base_transition_outputs = [
    "//command_line_option:apple configuration distinguisher",
    "//command_line_option:apple_platform_type",
    "//command_line_option:apple_split_cpu",
    "//command_line_option:compiler",
    "//command_line_option:cpu",
    "//command_line_option:crosstool_top",
    "//command_line_option:platforms",
    "//command_line_option:fission",
    "//command_line_option:grte_top",
    "//command_line_option:ios_minimum_os",
    "//command_line_option:macos_minimum_os",
    "//command_line_option:tvos_minimum_os",
    "//command_line_option:watchos_minimum_os",
]

apple_crosstool_transition = transition(
    implementation = _apple_crosstool_transition_impl,
    inputs = _apple_rule_base_transition_inputs,
    outputs = _apple_rule_base_transition_outputs,
)

objc_library = rule(
    implementation = _objc_library_impl,
    attrs = common_attrs.union(
        {
            "data": attr.label_list(allow_files = True),
            "_cc_toolchain": attr.label(
                default = "@" + semantics.get_repo() + "//tools/cpp:current_cc_toolchain",
            ),
        },
        common_attrs.LICENSES,
        common_attrs.COMPILING_RULE,
        common_attrs.COMPILE_DEPENDENCY_RULE,
        common_attrs.INCLUDE_SCANNING_RULE,
        common_attrs.SDK_FRAMEWORK_DEPENDER_RULE,
        common_attrs.COPTS_RULE,
        common_attrs.X_C_RUNE_RULE,
    ),
    fragments = ["objc", "apple", "cpp"],
    cfg = apple_crosstool_transition,
    toolchains = ["@" + semantics.get_repo() + "//tools/cpp:toolchain_type"],
    incompatible_use_toolchain_transition = True,
)
