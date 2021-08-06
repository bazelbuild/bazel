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

objc_internal = _builtins.internal.objc_internal
CcInfo = _builtins.toplevel.CcInfo
cc_common = _builtins.toplevel.cc_common
transition = _builtins.toplevel.transition
coverage_common = _builtins.toplevel.coverage_common
apple_common = _builtins.toplevel.apple_common

def _apple_static_library_impl(ctx):
    _validate_minimum_os(ctx)
    if not hasattr(apple_common.platform_type, ctx.attr.platform_type):
        fail("Unsupported platform type \"{}\"".format(ctx.attr.platform_type))
    platform_type = getattr(apple_common.platform_type, ctx.attr.platform_type)
    platform = ctx.fragments.apple.multi_arch_platform(platform_type)

    cpu_to_deps_map = ctx.split_attr.deps
    cpu_to_avoid_deps_map = ctx.split_attr.avoid_deps
    child_configs_and_toolchains = ctx.split_attr._child_configuration_dummmy
    rule_intermediate_artifacts = objc_internal.create_intermediate_artifacts(ctx = ctx)

    libraries_to_lipo = []
    files_to_build = [rule_intermediate_artifacts.combined_architecture_archive]
    sdk_dylib = []
    sdk_framework = []
    weak_sdk_framework = []

    for key, child_toolchain in ctx.split_attr._child_configuration_dummmy.items():
        intermediate_artifacts = objc_internal.create_intermediate_artifacts(ctx = ctx)

        deps = cpu_to_deps_map[key]

        common_variables = compilation_support.build_common_variables(
            ctx = ctx,
            toolchain = child_toolchain[cc_common.CcToolchainInfo],
            use_pch = True,
            deps = cpu_to_deps_map[key],
        )

        avoid_objc_providers = []
        avoid_cc_providers = []

        if len(cpu_to_avoid_deps_map.keys()):
            for dep in cpu_to_avoid_deps_map[key]:
                if apple_common.Objc in dep:
                    avoid_objc_providers.append(dep[apple_common.Objc])
                if CcInfo in dep:
                    avoid_cc_providers.append(dep[CcInfo])

        objc_provider = common_variables.objc_provider.subtract_subtrees(avoid_objc_providers = avoid_objc_providers, avoid_cc_providers = avoid_cc_providers)

        name = ctx.label.name + "-" + key + "-fl"

        linking_outputs = compilation_support.register_fully_link_action(common_variables, objc_provider, name)

        libraries_to_lipo.append(linking_outputs.library_to_link.static_library)

        sdk_dylib.append(objc_provider.sdk_dylib)
        sdk_framework.append(objc_provider.sdk_framework)
        weak_sdk_framework.append(objc_provider.weak_sdk_framework)

    objc_provider = apple_common.new_objc_provider(
        sdk_dylib = depset(transitive = sdk_dylib),
        sdk_framework = depset(transitive = sdk_framework),
        weak_sdk_framework = depset(transitive = weak_sdk_framework),
    )

    header_tokens = []
    for key, deps in cpu_to_deps_map.items():
        for dep in deps:
            if CcInfo in dep:
                header_tokens.append(dep[CcInfo].compilation_context.validation_artifacts)

    output_groups = {"_validation": depset(transitive = header_tokens)}

    output_archive = rule_intermediate_artifacts.combined_architecture_archive
    _register_combine_architectures_action(
        ctx,
        libraries_to_lipo,
        output_archive,
        platform,
    )

    runfiles = ctx.runfiles(files = files_to_build, collect_default = True, collect_data = True)

    return [
        DefaultInfo(files = depset(files_to_build), runfiles = runfiles),
        objc_provider,
        OutputGroupInfo(**output_groups),
        apple_common.AppleStaticLibrary(archive = output_archive, objc = objc_provider),
    ]

def _validate_minimum_os(ctx):
    if len(ctx.attr.minimum_os_version) == 0:
        if ctx.fragments.apple.mandatory_minimum_version():
            fail("'minimum_os_version' must be explicitly specified")
        else:
            return

    minimum_os_version = apple_common.dotted_version(ctx.attr.minimum_os_version)
    components = ctx.attr.minimum_os_version.split(".")
    for i in range(len(components) - 1, 0, -1):
        if components[i] == "0":
            components.pop()
        else:
            break

    if len(components) > 2:
        fail("Invalid version string. Cannot have more than two components")
    for component in components:
        if not component.isdigit():
            fail("Invalid version string. Must be numeric")

def _register_combine_architectures_action(ctx, artifacts, output_binary, platform):
    if len(artifacts) > 1:
        xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]
        env = {}
        env.update(apple_common.apple_host_system_env(xcode_config))
        env.update(apple_common.target_apple_env(xcode_config, platform))

        execution_requirements = {}
        execution_requirements.update(xcode_config.execution_info())

        arguments = ctx.actions.args()
        arguments.add("lipo")
        arguments.add("-create")
        arguments.add_all(artifacts)
        arguments.add("-o")
        arguments.add(output_binary)
        ctx.actions.run(
            mnemonic = "ObjcCombiningArchitectures",
            inputs = artifacts,
            outputs = [output_binary],
            env = env,
            execution_requirements = execution_requirements,
            arguments = [arguments],
            executable = ctx.executable._xcrunwrapper,
        )
    else:
        ctx.actions.symlink(
            output = output_binary,
            target_file = artifacts[0],
        )

apple_static_library = rule(
    implementation = _apple_static_library_impl,
    attrs = common_attrs.union(
        {
            "data": attr.label_list(allow_files = True),
            "deps": attr.label_list(
                cfg = apple_common.multi_arch_split,
                providers = [apple_common.Objc],
                flags = ["DIRECT_COMPILE_TIME_INPUT"],
                allow_rules = ["cc_library", "cc_inc_library"],
            ),
            "avoid_deps": attr.label_list(
                cfg = apple_common.multi_arch_split,
                providers = [apple_common.Objc],
                flags = ["DIRECT_COMPILE_TIME_INPUT"],
                allow_rules = ["cc_library", "cc_inc_library"],
            ),
            "linkopts": attr.string_list(),
            "additional_linker_inputs": attr.label_list(flags = ["DIRECT_COMPILE_TIME_INPUT"], allow_files = True),
            "_child_configuration_dummmy": attr.label(
                cfg = apple_common.multi_arch_split,
                default = "@" + semantics.get_repo() + "//tools/cpp:current_cc_toolchain",
            ),
            "_cc_toolchain": attr.label(
                default = "@" + semantics.get_repo() + "//tools/cpp:current_cc_toolchain",
            ),
        },
        common_attrs.LICENSES,
        common_attrs.X_C_RUNE_RULE,
        common_attrs.SDK_FRAMEWORK_DEPENDER_RULE,
        common_attrs.PLATFORM_RULE,
    ),
    outputs = {
        "lipo_archive": "%{name}_lipo.a",
    },
    cfg = apple_crosstool_transition,
    fragments = ["objc", "apple", "cpp"],
    toolchains = ["@" + semantics.get_repo() + "//tools/cpp:toolchain_type"],
    incompatible_use_toolchain_transition = True,
)
