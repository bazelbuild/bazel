# Copyright 2025 The Bazel Authors. All rights reserved.
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
"""
Holds information collected for .o bitcode files coming from a ThinLTO C(++) compilation.
"""

load(":common/cc/cc_helper_internal.bzl", _PRIVATE_STARLARKIFICATION_ALLOWLIST = "PRIVATE_STARLARKIFICATION_ALLOWLIST")

_cc_internal = _builtins.internal.cc_internal
_cc_common_internal = _builtins.internal.cc_common

LtoCompilationContextInfo = provider(
    doc = """
Holds information collected for .o bitcode files coming from a ThinLTO C(++) compilation.
Specifically, maps each bitcode file to the corresponding minimized bitcode file
that can be used for the LTO indexing step, as well as to compile flags applying to that
compilation that should also be applied to the LTO backend compilation invocation.
""",
    fields = {
        "lto_bitcode_inputs": "(dict[File, BitcodeInfo]) Maps each bitcode file to the corresponding minimized bitcode file and compile flags.",
    },
)

BitcodeInfo = provider(
    doc = """Holds information about a bitcode file produced by the compile action needed by
             the LTO indexing and backend actions.""",
    fields = {
        "minimized_bitcode": "(File) The minimized bitcode file produced by the compile and used by LTO indexing.",
        "copts": "(list[str]) The compiler flags used for the compile that should also be used when finishing compilation during the LTO backend.",
    },
)

# IMPORTANT: This function is public API exposed on cc_common module!
def create_lto_compilation_context(*, objects = {}):
    """Creates an LtoCompilationContextInfo provider.

    Args:
        objects: (dict[File, tuple[File, list[str]]]) A map of full object to index object and copts.

    Returns:
        An LtoCompilationContextInfo provider.
    """
    _cc_common_internal.check_private_api(allowlist = _PRIVATE_STARLARKIFICATION_ALLOWLIST)
    bitcode_infos = {}
    for k, (minimized_bitcode, copts) in objects.items():
        if type(minimized_bitcode) != "File" and minimized_bitcode != None:
            fail("expected Artifact for minimized bitcode, got " + type(minimized_bitcode))
        if type(copts) != "list":
            fail("expected list for copts, got " + type(copts))
        bitcode_infos[k] = BitcodeInfo(minimized_bitcode = minimized_bitcode, copts = copts)
    if not bitcode_infos:
        return EMPTY_LTO_COMPILATION_CONTEXT
    return LtoCompilationContextInfo(lto_bitcode_inputs = _cc_internal.freeze(bitcode_infos))

def merge_lto_compilation_contexts(*, lto_compilation_contexts):
    """Merges a list of LtoCompilationContextInfo providers.

    Args:
        lto_compilation_contexts: (list[LtoCompilationContextInfo]) The LtoCompilationContextInfo providers to merge.

    Returns:
        An LtoCompilationContextInfo provider with the merged information.
    """
    if not lto_compilation_contexts:
        return EMPTY_LTO_COMPILATION_CONTEXT
    if len(lto_compilation_contexts) == 1:
        return lto_compilation_contexts[0]
    bitcode_infos = {}
    for lto_compilation_context in lto_compilation_contexts:
        bitcode_infos.update(lto_compilation_context.lto_bitcode_inputs)
    return LtoCompilationContextInfo(lto_bitcode_inputs = _cc_internal.freeze(bitcode_infos))

def get_minimized_bitcode_or_self(lto_compilation_context, full_bitcode):
    """Gets the minimized bitcode corresponding to the full bitcode file, or returns full bitcode if it doesn't exist.

    Args:
        lto_compilation_context: (LtoCompilationContextInfo) The LtoCompilationContextInfo provider.
        full_bitcode: (File) The full bitcode file.

    Returns:
        (File) The minimized bitcode file or the full bitcode file.
    """
    bitcode_info = lto_compilation_context.lto_bitcode_inputs.get(full_bitcode)
    if bitcode_info == None or bitcode_info.minimized_bitcode == None:
        return full_bitcode
    return bitcode_info.minimized_bitcode

EMPTY_LTO_COMPILATION_CONTEXT = LtoCompilationContextInfo(lto_bitcode_inputs = {})
