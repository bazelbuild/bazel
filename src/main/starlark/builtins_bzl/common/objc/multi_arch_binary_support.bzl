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
load(":common/objc/compilation_support.bzl", "compilation_support")

def _list_or_depset_to_list(list_or_depset):
    if type(list_or_depset) == type([]):
        return list_or_depset
    return list_or_depset.to_list()

def _build_avoid_library_set(avoid_dep_linking_contexts):
    avoid_library_set = dict()
    for linking_context in avoid_dep_linking_contexts:
        for linker_input in linking_context.linker_inputs.to_list():
            for library_to_link in linker_input.libraries:
                library = compilation_support.get_static_library_for_linking(library_to_link)
                if library != None:
                    avoid_library_set[library] = True
    return avoid_library_set

def subtract_linking_contexts(ctx, linking_contexts, avoid_dep_linking_contexts):
    """Subtracts the libraries in avoid_dep_linking_contexts from linking_contexts.

    Args:
      ctx: The rule context.
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
        for linker_input in _list_or_depset_to_list(linking_context.linker_inputs):
            for library_to_link in _list_or_depset_to_list(linker_input.libraries):
                library = library_to_link.static_library
                if library not in avoid_library_set:
                    libraries.append(library_to_link)
            user_link_flags.extend(linker_input.user_link_flags)
            additional_inputs.extend(linker_input.additional_inputs)
            linkstamps.extend(linker_input.linkstamps)
    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        libraries = depset(libraries, order = "topological"),
        user_link_flags = user_link_flags,
        additional_inputs = depset(additional_inputs),
        linkstamps = depset(linkstamps),
    )
    return cc_common.create_linking_context(linker_inputs = depset([linker_input]))
