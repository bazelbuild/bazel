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
