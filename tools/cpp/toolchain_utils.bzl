# pylint: disable=g-bad-file-header
# Copyright 2016 The Bazel Authors. All rights reserved.
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
Finds the c++ toolchain.

Returns the toolchain if enabled, and falls back to a toolchain constructed from
the CppConfiguration.
"""

def find_cpp_toolchain(ctx):
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
        if "@bazel_tools//tools/cpp:toolchain_type" in ctx.toolchains:
            return ctx.toolchains["@bazel_tools//tools/cpp:toolchain_type"]
        fail("In order to use find_cpp_toolchain, you must include the '@bazel_tools//tools/cpp:toolchain_type' in the toolchains argument to your rule.")

    # Fall back to the legacy implicit attribute lookup.
    if hasattr(ctx.attr, "_cc_toolchain"):
        return ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]

    # We didn't find anything.
    fail("In order to use find_cpp_toolchain, you must define the '_cc_toolchain' attribute on your rule or aspect.")
