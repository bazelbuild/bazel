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
Utilities to help work with c++ toolchains.
"""

CPP_TOOLCHAIN_TYPE = "@bazel_tools//tools/cpp:toolchain_type"

def find_cpp_toolchain(ctx, *, mandatory = True):
    """
    Finds the c++ toolchain.

    If the c++ toolchain is in use, returns it.  Otherwise, returns a c++
    toolchain derived from legacy toolchain selection, constructed from
    the CppConfiguration.

    Args:
      ctx: The rule context for which to find a toolchain.
      mandatory: If this is set to False, this function will return None rather
        than fail if no toolchain is found. To use this parameter, the calling
        rule should have a `_cc_toolchain` label attribute with default
        `@bazel_tools//tools/cpp:optional_current_cc_toolchain`.

    Returns:
      A CcToolchainProvider, or None if the c++ toolchain is declared as
      optional, mandatory is False and no toolchain has been found.
    """

    # Check the incompatible flag for toolchain resolution.
    if hasattr(cc_common, "is_cc_toolchain_resolution_enabled_do_not_use") and cc_common.is_cc_toolchain_resolution_enabled_do_not_use(ctx = ctx):
        if not CPP_TOOLCHAIN_TYPE in ctx.toolchains:
            fail("In order to use find_cpp_toolchain, you must include the '%s' in the toolchains argument to your rule." % CPP_TOOLCHAIN_TYPE)
        toolchain_info = ctx.toolchains[CPP_TOOLCHAIN_TYPE]
        if toolchain_info == None:
            if not mandatory:
                return None

            # No cpp toolchain was found, so report an error.
            fail("Unable to find a CC toolchain using toolchain resolution. Target: %s, Platform: %s, Exec platform: %s" %
                 (ctx.label, ctx.fragments.platform.platform, ctx.fragments.platform.host_platform))
        if hasattr(toolchain_info, "cc_provider_in_toolchain") and hasattr(toolchain_info, "cc"):
            return toolchain_info.cc
        return toolchain_info

    # Fall back to the legacy implicit attribute lookup.
    if hasattr(ctx.attr, "_cc_toolchain"):
        return ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]

    # We didn't find anything.
    fail("In order to use find_cpp_toolchain, you must define the '_cc_toolchain' attribute on your rule or aspect.")

def use_cpp_toolchain(mandatory = False):
    """
    Helper to depend on the c++ toolchain.

    Usage:
    ```
    my_rule = rule(
        toolchains = [other toolchain types] + use_cpp_toolchain(),
    )
    ```

    Args:
      mandatory: Whether or not it should be an error if the toolchain cannot be resolved.

    Returns:
      A list that can be used as the value for `rule.toolchains`.
    """
    return [config_common.toolchain_type(CPP_TOOLCHAIN_TYPE, mandatory = mandatory)]
