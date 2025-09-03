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

CcInfo = _builtins.toplevel.CcInfo

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
        files = depset(compilation_outputs.dwo_files()),
        pic_files = depset(compilation_outputs.pic_dwo_files()),
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
