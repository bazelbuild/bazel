# Copyright 2024 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License(**kwargs): Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing(**kwargs): software
# distributed under the License is distributed on an "AS IS" BASIS(**kwargs):
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND(**kwargs): either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Support utility for creating multi-arch Apple binaries."""

load(":common/cc/cc_common.bzl", "cc_common")
load(":common/objc/apple_platform.bzl", "apple_platform")
load(":common/objc/compilation_support.bzl", "compilation_support")

objc_internal = _builtins.internal.objc_internal
TargetTripletInfo = provider(
    "Contains the the target triplet (architecture, platform, environment) for a given configuration.",
    fields = {
        "architecture": "string, the CPU as returned by AppleConfiguration.getSingleArchitecture()",
        "platform": "apple_platform.PLATFORM_TPYE string as returned by apple_platform.get_target_platform()",
        "environment": "string ('device', 'simulator' or 'macabi) as returned by apple_platform.get_target_environment",
    },
)

def _build_avoid_library_set(avoid_dep_linking_contexts):
    avoid_library_set = dict()
    for linking_context in avoid_dep_linking_contexts:
        for linker_input in linking_context.linker_inputs.to_list():
            for library_to_link in linker_input.libraries:
                library_artifact = compilation_support.get_static_library_for_linking(library_to_link)
                if library_artifact:
                    avoid_library_set[library_artifact.short_path] = True
    return avoid_library_set

def subtract_linking_contexts(owner, linking_contexts, avoid_dep_linking_contexts):
    """Subtracts the libraries in avoid_dep_linking_contexts from linking_contexts.

    Args:
      owner: The label of the target currently being analyzed.
      linking_contexts: An iterable of CcLinkingContext objects.
      avoid_dep_linking_contexts: An iterable of CcLinkingContext objects.

    Returns:
      A CcLinkingContext object.
    """
    libraries = []
    user_link_flags = []
    additional_inputs = []
    linkstamps = []
    avoid_library_set = _build_avoid_library_set(avoid_dep_linking_contexts)
    for linking_context in linking_contexts:
        for linker_input in linking_context.linker_inputs.to_list():
            for library_to_link in linker_input.libraries:
                library_artifact = compilation_support.get_library_for_linking(library_to_link)
                if library_artifact.short_path not in avoid_library_set:
                    libraries.append(library_to_link)
            user_link_flags.extend(linker_input.user_link_flags)
            additional_inputs.extend(linker_input.additional_inputs)
            linkstamps.extend(linker_input.linkstamps)
    linker_input = cc_common.create_linker_input(
        owner = owner,
        libraries = depset(libraries, order = "topological"),
        user_link_flags = user_link_flags,
        additional_inputs = depset(additional_inputs),
        linkstamps = depset(linkstamps),
    )
    return cc_common.create_linking_context(
        linker_inputs = depset([linker_input]),
        owner = owner,
    )

def _get_target_triplet(config):
    """Returns the target triplet (architecture, platform, environment) for a given configuration."""
    cpu_platform = apple_platform.for_target_cpu(objc_internal.get_cpu(config))
    apple_config = objc_internal.get_apple_config(config)

    return TargetTripletInfo(
        architecture = apple_config.single_arch_cpu,
        platform = apple_platform.get_target_platform(cpu_platform),
        environment = apple_platform.get_target_environment(cpu_platform),
    )

def get_split_target_triplet(ctx):
    """Transforms a rule context's ctads to a Starlark Dict mapping transitions to target triplets.

    Args:
      ctx: The Starlark rule context.

    Returns:
      A Starlark Dict<String, StructImpl> keyed by split transition keys with
      their target triplet (architecture, platform, environment) as value.
    """
    result = dict()
    ctads = objc_internal.get_split_prerequisites(ctx)
    for split_transition_key, config in ctads.items():
        if split_transition_key == None:
            fail("unexpected empty key in split transition")
        result[split_transition_key] = _get_target_triplet(config)
    return result
