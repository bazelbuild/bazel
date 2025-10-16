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
# limitations under the License
"""
Definition of CcInfo provider.
"""

load(":common/cc/cc_helper_internal.bzl", "check_private_api")
load(":common/cc/link/create_extra_link_time_library.bzl", "create_extra_link_time_libraries", "merge_extra_link_time_libraries")

_cc_common_internal = _builtins.internal.cc_common
_cc_internal = _builtins.internal.cc_internal

CcLinkingContextInfo = provider(
    "CcLinkingContextInfo",
    fields = {
        "linker_inputs": "A depset of linker inputs.",
        "_extra_link_time_libraries": "Extra link time libraries.",
    },
)

CcNativeLibraryInfo = provider(
    "CcNativeLibraryInfo",
    fields = ["libraries_to_link"],
)

CcDebugContextInfo = provider(
    doc = """
        C++ debug related objects, specifically when fission is used.
        Stores .dwo files which can be combined into a .dwp in the packaging step.

        <p>It is not expected for this to be used externally at this time. This API is experimental
        and subject to change, and its usage should be restricted to internal packages.
    """,
    fields = {
        "files": """(.dwo files) The .dwo files for non-PIC compilation.
            Returns the .dwo files that should be included in this target's .dwp packaging (if this
            target is linked) or passed through to a dependant's .dwp packaging (e.g. if this is a
            cc_library depended on by a statically linked cc_binary).
            Assumes the corresponding link consumes .o files (vs. .pic.o files).
            """,
        "pic_files": "(.dwo files) The .dwo files for PIC compilation.",
    },
)

_EMPTY_LINKING_CONTEXT = CcLinkingContextInfo(
    linker_inputs = depset(),
    _extra_link_time_libraries = None,
)

_EMPTY_DEBUG_CONTEXT = CcDebugContextInfo(
    files = depset(),
    pic_files = depset(),
)

def create_linking_context(
        *,
        linker_inputs,
        extra_link_time_library = None):
    """Creates a CcLinkingContextInfo provider.

    Args:
        linker_inputs: A depset of linker inputs.
        extra_link_time_library: An optional extra link time library.

    Returns:
        A CcLinkingContextInfo provider.
    """
    return CcLinkingContextInfo(
        linker_inputs = linker_inputs,
        _extra_link_time_libraries = create_extra_link_time_libraries(extra_link_time_library),
    )

def merge_linking_contexts(*, linking_contexts):
    """Merges a list of CcLinkingContextInfo providers.
    """
    linker_inputs = depset(transitive = [ctx.linker_inputs for ctx in linking_contexts], order = "topological")
    extra_link_time_libraries = merge_extra_link_time_libraries([ctx._extra_link_time_libraries for ctx in linking_contexts if ctx._extra_link_time_libraries != None])
    return CcLinkingContextInfo(
        linker_inputs = linker_inputs,
        _extra_link_time_libraries = extra_link_time_libraries,
    )

def create_debug_context(compilation_outputs):
    """Creates a CcDebugContextInfo from CcCompilationOutputs.

    Args:
        compilation_outputs: A CcCompilationOutputs object.

    Returns:
        A new CcDebugContextInfo object.
    """
    check_private_api()
    return CcDebugContextInfo(
        files = depset(compilation_outputs._dwo_files),
        pic_files = depset(compilation_outputs._pic_dwo_files),
    )

def merge_debug_context(debug_contexts = []):
    """Merge multiple CcDebugContextInfos into one.

    Args:
        debug_contexts: A list of CcDebugContextInfo objects.

    Returns:
        A new CcDebugContextInfo object.
    """
    check_private_api()
    if not debug_contexts:
        return _EMPTY_DEBUG_CONTEXT

    transitive_dwo_files = []
    transitive_pic_dwo_files = []

    for ctx in debug_contexts:
        transitive_dwo_files.append(ctx.files)
        transitive_pic_dwo_files.append(ctx.pic_files)

    return CcDebugContextInfo(
        files = depset(transitive = transitive_dwo_files),
        pic_files = depset(transitive = transitive_pic_dwo_files),
    )

def _create_cc_info(
        *,
        compilation_context = None,
        linking_context = None,
        debug_context = None,
        cc_native_library_info = None):
    return dict(
        compilation_context = compilation_context or _cc_internal.empty_compilation_context(),
        linking_context = linking_context or _EMPTY_LINKING_CONTEXT,
        _debug_context = debug_context or _EMPTY_DEBUG_CONTEXT,
        _legacy_transitive_native_libraries = cc_native_library_info.libraries_to_link if cc_native_library_info else depset(),
    )

CcInfo, _ = provider(
    doc = "Provider for C++ compilation and linking information.",
    fields = {
        "compilation_context": "A `CcCompilationContext`.",
        "linking_context": "A `CcLinkingContext`.",
        "_debug_context": "A `CcDebugInfoContext`.",
        "_legacy_transitive_native_libraries": "A `CcNativeLibraryInfo`.",
    },
    init = _create_cc_info,
)

def merge_cc_infos(*, direct_cc_infos = [], cc_infos = []):
    """
    Merges multiple `CcInfo`s into one.

    Args:
      direct_cc_infos: List of `CcInfo`s to be merged, whose headers will be exported by
        the direct fields in the returned provider.
      cc_infos: List of `CcInfo`s to be merged, whose headers will not be exported
        by the direct fields in the returned provider.

    Returns:
      Merged CcInfo.
    """
    direct_cc_compilation_contexts = []
    cc_compilation_contexts = []
    cc_linking_contexts = []
    cc_debug_info_contexts = []
    transitive_native_cc_libraries = []

    for cc_info in direct_cc_infos:
        direct_cc_compilation_contexts.append(cc_info.compilation_context)
        cc_linking_contexts.append(cc_info.linking_context)
        cc_debug_info_contexts.append(cc_info._debug_context)
        transitive_native_cc_libraries.append(cc_info._legacy_transitive_native_libraries)

    for cc_info in cc_infos:
        cc_compilation_contexts.append(cc_info.compilation_context)
        cc_linking_contexts.append(cc_info.linking_context)
        cc_debug_info_contexts.append(cc_info._debug_context)
        transitive_native_cc_libraries.append(cc_info._legacy_transitive_native_libraries)

    return CcInfo(
        compilation_context = _cc_common_internal.merge_compilation_contexts(compilation_contexts = direct_cc_compilation_contexts, non_exported_compilation_contexts = cc_compilation_contexts),
        linking_context = merge_linking_contexts(linking_contexts = cc_linking_contexts),
        debug_context = merge_debug_context(cc_debug_info_contexts),
        cc_native_library_info = CcNativeLibraryInfo(libraries_to_link = depset(order = "topological", transitive = transitive_native_cc_libraries)),
    )
