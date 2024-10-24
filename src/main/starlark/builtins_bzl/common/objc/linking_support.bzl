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

"""apple_common.link_multi_arch_static_library Starlark implementation"""

load(":common/cc/cc_common.bzl", "cc_common")
load(":common/cc/cc_info.bzl", "CcInfo")
load(":common/objc/compilation_support.bzl", "compilation_support")
load(":common/objc/multi_arch_binary_support.bzl", "get_split_target_triplet", "subtract_linking_contexts")
load(":common/objc/objc_info.bzl", "ObjcInfo")

objc_internal = _builtins.internal.objc_internal

AppleDynamicFrameworkInfo = provider(
    doc = "Contains information about an Apple dynamic framework.",
    fields = {
        "framework_dirs": """\
The framework path names used as link inputs in order to link against the
dynamic framework.
""",
        "framework_files": """\
The full set of artifacts that should be included as inputs to link against the
dynamic framework.
""",
        "binary": "The dylib binary artifact of the dynamic framework.",
        "cc_info": """\
A `CcInfo` which contains information about the transitive dependencies linked
into the binary.
""",
    },
)

AppleExecutableBinaryInfo = provider(
    doc = """
Contains the executable binary output that was built using
`link_multi_arch_binary` with the `executable` binary type.
""",
    fields = {
        "binary": """\
The executable binary artifact output by `link_multi_arch_binary`.
""",
        "cc_info": """\
A `CcInfo` which contains information about the transitive dependencies linked
into the binary.
""",
    },
)

AppleDebugOutputsInfo = provider(
    """
Holds debug outputs of an Apple binary rule.

The only field is `output_map`, which is a dictionary of:
  `{ arch: { "dsym_binary": File, "linkmap": File }`

Where `arch` is any Apple architecture such as "arm64" or "armv7".
""",
    fields = ["outputs_map"],
)

def _link_multi_arch_static_library(ctx):
    """Links a (potentially multi-architecture) static library targeting Apple platforms.

    Rule context is a required parameter due to usage of the cc_common.configure_features API.

    Args:
        ctx: The Starlark rule context.

    Returns:
        A Starlark struct containing the following attributes:
            - output_groups: OutputGroupInfo provider from transitive CcInfo validation_artifacts.
            - outputs: List of structs containing the following attributes:
                - library: Artifact representing a linked static library.
                - architecture: Linked static library architecture (e.g. 'arm64', 'x86_64').
                - platform: Linked static library target Apple platform (e.g. 'ios', 'macos').
                - environment: Linked static library environment (e.g. 'device', 'simulator').
    """

    split_target_triplets = get_split_target_triplet(ctx)

    split_deps = ctx.split_attr.deps
    split_avoid_deps = ctx.split_attr.avoid_deps
    child_configs_and_toolchains = ctx.split_attr._child_configuration_dummy

    outputs = []

    for split_transition_key, child_toolchain in child_configs_and_toolchains.items():
        cc_toolchain = child_toolchain[cc_common.CcToolchainInfo]
        common_variables = compilation_support.build_common_variables(
            ctx = ctx,
            toolchain = cc_toolchain,
            use_pch = True,
            deps = split_deps[split_transition_key],
        )

        avoid_objc_providers = []
        avoid_cc_providers = []
        avoid_cc_linking_contexts = []

        if len(split_avoid_deps.keys()):
            for dep in split_avoid_deps[split_transition_key]:
                if ObjcInfo in dep:
                    avoid_objc_providers.append(dep[ObjcInfo])
                if CcInfo in dep:
                    avoid_cc_providers.append(dep[CcInfo])
                    avoid_cc_linking_contexts.append(dep[CcInfo].linking_context)

        name = ctx.label.name + "-" + cc_toolchain.target_gnu_system_name + "-fl"

        cc_linking_context = subtract_linking_contexts(
            owner = ctx.label,
            linking_contexts = common_variables.objc_linking_context.cc_linking_contexts,
            avoid_dep_linking_contexts = avoid_cc_linking_contexts,
        )
        linking_outputs = compilation_support.register_fully_link_action(
            name = name,
            common_variables = common_variables,
            cc_linking_context = cc_linking_context,
        )

        output = {
            "library": linking_outputs.library_to_link.static_library,
        }

        if split_target_triplets != None:
            target_triplet = split_target_triplets.get(split_transition_key)
            output["platform"] = target_triplet.platform
            output["architecture"] = target_triplet.architecture
            output["environment"] = target_triplet.environment

        outputs.append(struct(**output))

    header_tokens = []
    for _, deps in split_deps.items():
        for dep in deps:
            if CcInfo in dep:
                header_tokens.append(dep[CcInfo].compilation_context.validation_artifacts)

    output_groups = {"_validation": depset(transitive = header_tokens)}

    return struct(
        outputs = outputs,
        output_groups = OutputGroupInfo(**output_groups),
    )

def _link_multi_arch_binary(
        *,
        ctx,
        avoid_deps = [],
        extra_linkopts = [],
        extra_link_inputs = [],
        extra_requested_features = [],
        extra_disabled_features = [],
        stamp = -1,
        variables_extension = {}):
    """Links a (potentially multi-architecture) binary targeting Apple platforms.

    This method comprises a bulk of the logic of the Starlark <code>apple_binary</code>
    rule in the rules_apple domain and exists to aid in the migration of its
    linking logic to Starlark in rules_apple.

    This API is **highly experimental** and subject to change at any time. Do
    not depend on the stability of this function at this time.
    """

    split_target_triplets = get_split_target_triplet(ctx)
    split_build_configs = objc_internal.get_split_build_configs(ctx)
    split_deps = ctx.split_attr.deps
    child_configs_and_toolchains = ctx.split_attr._child_configuration_dummy

    if split_deps and split_deps.keys() != child_configs_and_toolchains.keys():
        fail(("Split transition keys are different between 'deps' [%s] and " +
              "'_child_configuration_dummy' [%s]") % (
            split_deps.keys(),
            child_configs_and_toolchains.keys(),
        ))

    avoid_cc_infos = [
        dep[AppleDynamicFrameworkInfo].cc_info
        for dep in avoid_deps
        if AppleDynamicFrameworkInfo in dep
    ]
    avoid_cc_infos.extend([
        dep[AppleExecutableBinaryInfo].cc_info
        for dep in avoid_deps
        if AppleExecutableBinaryInfo in dep
    ])
    avoid_cc_infos.extend([dep[CcInfo] for dep in avoid_deps if CcInfo in dep])
    avoid_cc_linking_contexts = [dep.linking_context for dep in avoid_cc_infos]

    outputs = []
    cc_infos = []
    legacy_debug_outputs = {}

    cc_infos.extend(avoid_cc_infos)

    # $(location...) is only used in one test, and tokenize only affects linkopts in one target
    additional_linker_inputs = getattr(ctx.attr, "additional_linker_inputs", [])
    attr_linkopts = [
        ctx.expand_location(opt, targets = additional_linker_inputs)
        for opt in getattr(ctx.attr, "linkopts", [])
    ]
    attr_linkopts = [token for opt in attr_linkopts for token in ctx.tokenize(opt)]

    for split_transition_key, child_toolchain in child_configs_and_toolchains.items():
        cc_toolchain = child_toolchain[cc_common.CcToolchainInfo]
        deps = split_deps.get(split_transition_key, [])
        target_triplet = split_target_triplets.get(split_transition_key)

        common_variables = compilation_support.build_common_variables(
            ctx = ctx,
            toolchain = cc_toolchain,
            deps = deps,
            extra_disabled_features = extra_disabled_features,
            extra_enabled_features = extra_requested_features,
            attr_linkopts = attr_linkopts,
        )

        cc_infos.append(CcInfo(
            compilation_context = cc_common.merge_compilation_contexts(
                compilation_contexts =
                    common_variables.objc_compilation_context.cc_compilation_contexts,
            ),
            linking_context = cc_common.merge_linking_contexts(
                linking_contexts = common_variables.objc_linking_context.cc_linking_contexts,
            ),
        ))

        cc_linking_context = subtract_linking_contexts(
            owner = ctx.label,
            linking_contexts = common_variables.objc_linking_context.cc_linking_contexts +
                               avoid_cc_linking_contexts,
            avoid_dep_linking_contexts = avoid_cc_linking_contexts,
        )

        child_config = split_build_configs.get(split_transition_key)

        additional_outputs = []
        extensions = {}

        dsym_binary = None
        if ctx.fragments.cpp.apple_generate_dsym:
            if ctx.fragments.cpp.objc_should_strip_binary:
                suffix = "_bin_unstripped.dwarf"
            else:
                suffix = "_bin.dwarf"
            dsym_binary = ctx.actions.declare_shareable_artifact(
                ctx.label.package + "/" + ctx.label.name + suffix,
                child_config.bin_dir,
            )
            extensions["dsym_path"] = dsym_binary.path  # dsym symbol file
            additional_outputs.append(dsym_binary)
            legacy_debug_outputs.setdefault(target_triplet.architecture, {})["dsym_binary"] = dsym_binary

        linkmap = None
        if ctx.fragments.cpp.objc_generate_linkmap:
            linkmap = ctx.actions.declare_shareable_artifact(
                ctx.label.package + "/" + ctx.label.name + ".linkmap",
                child_config.bin_dir,
            )
            extensions["linkmap_exec_path"] = linkmap.path  # linkmap file
            additional_outputs.append(linkmap)
            legacy_debug_outputs.setdefault(target_triplet.architecture, {})["linkmap"] = linkmap

        name = ctx.label.name + "_bin"
        executable = compilation_support.register_configuration_specific_link_actions(
            name = name,
            common_variables = common_variables,
            cc_linking_context = cc_linking_context,
            build_config = child_config,
            extra_link_args = extra_linkopts,
            stamp = stamp,
            user_variable_extensions = variables_extension | extensions,
            additional_outputs = additional_outputs,
            deps = deps,
            extra_link_inputs = extra_link_inputs,
            attr_linkopts = attr_linkopts,
        )

        output = {
            "binary": executable,
            "platform": target_triplet.platform,
            "architecture": target_triplet.architecture,
            "environment": target_triplet.environment,
            "dsym_binary": dsym_binary,
            "linkmap": linkmap,
        }

        outputs.append(struct(**output))

    header_tokens = []
    for _, deps in split_deps.items():
        for dep in deps:
            if CcInfo in dep:
                header_tokens.append(dep[CcInfo].compilation_context.validation_artifacts)

    output_groups = {"_validation": depset(transitive = header_tokens)}

    return struct(
        cc_info = cc_common.merge_cc_infos(direct_cc_infos = cc_infos),
        output_groups = output_groups,
        outputs = outputs,
        debug_outputs_provider = AppleDebugOutputsInfo(outputs_map = legacy_debug_outputs),
    )

linking_support = struct(
    link_multi_arch_static_library = _link_multi_arch_static_library,
    link_multi_arch_binary = _link_multi_arch_binary,
)
