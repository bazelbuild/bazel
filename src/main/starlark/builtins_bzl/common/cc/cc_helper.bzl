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

"""Utility functions for C++ rules."""

load("@_builtins//:blaze/common/toplevel_aliases.bzl", "CcInfo", "cc_common")

def _merge_cc_debug_contexts(compilation_outputs, dep_cc_infos):
    debug_context = cc_common.create_debug_context(compilation_outputs)
    debug_contexts = [debug_context]
    for dep_cc_info in dep_cc_infos:
        debug_contexts.append(dep_cc_info.debug_context())

    return cc_common.merge_debug_context(debug_contexts)

def _is_code_coverage_enabled(ctx):
    if ctx.coverage_instrumented():
        return True
    for dep in ctx.attr.deps:
        if CcInfo in dep:
            if ctx.coverage_instrumented(dep):
                return True
    return False

def _get_dynamic_libraries_for_runtime(cc_linking_context, linking_statically):
    libraries = []
    for linker_input in cc_linking_context.linker_inputs.to_list():
        libraries.extend(linker_input.libraries)

    dynamic_libraries_for_runtime = []
    for library in libraries:
        artifact = _get_dynamic_library_for_runtime_or_none(library, linking_statically)
        if artifact != None:
            dynamic_libraries_for_runtime.append(artifact)

    return dynamic_libraries_for_runtime

def _get_dynamic_library_for_runtime_or_none(library, linking_statically):
    if library.dynamic_library == None:
        return None

    if linking_statically and (library.static_library != None or library.pic_static_library != None):
        return None

    return library.dynamic_library

def _find_cpp_toolchain(ctx):
    """
    Finds the c++ toolchain.

    If the c++ toolchain is in use, returns it.  Otherwise, returns a c++
    toolchain derived from legacy toolchain selection.

    Args:
      ctx: The rule context for which to find a toolchain.

    Returns:
      A CcToolchainProvider.
    """

    # Check the incompatible flag for toolchain resolution.
    if hasattr(cc_common, "is_cc_toolchain_resolution_enabled_do_not_use") and cc_common.is_cc_toolchain_resolution_enabled_do_not_use(ctx = ctx):
        if not "//tools/cpp:toolchain_type" in ctx.toolchains:
            fail("In order to use find_cpp_toolchain, you must include the '//tools/cpp:toolchain_type' in the toolchains argument to your rule.")
        return ctx.toolchains["//tools/cpp:toolchain_type"]

    # Otherwise, fall back to the legacy attribute.
    if hasattr(ctx.attr, "_cc_toolchain"):
        return ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]

    # We didn't find anything.
    fail("In order to use find_cpp_toolchain, you must define the '_cc_toolchain' attribute on your rule or aspect.")

cc_helper = struct(
    merge_cc_debug_contexts = _merge_cc_debug_contexts,
    is_code_coverage_enabled = _is_code_coverage_enabled,
    get_dynamic_libraries_for_runtime = _get_dynamic_libraries_for_runtime,
    get_dynamic_library_for_runtime_or_none = _get_dynamic_library_for_runtime_or_none,
    find_cpp_toolchain = _find_cpp_toolchain,
)
