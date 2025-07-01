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
The cc_common.create_extra_link_time_library function.

 * An extra library to include in a link. The actual library is built at link time.
 *
 * <p>This can be used for non-C++ inputs to a C++ link. A class that implements this interface will
 * support transitively gathering all inputs from link dependencies, and then combine them all
 * together into a set of C++ libraries.
 *
 * <p>Any implementations must be immutable (and therefore thread-safe), because this is passed
 * between rules and accessed in a multi-threaded context.
"""

cc_internal = _builtins.internal.cc_internal

# An implementation of ExtraLinkTimeLibrary that uses functions and data passed in from Starlark.

# Fields are intentionally not specified as they may be defined by users.
# Common fields are:
#  "build_library_func": "Starlark function to create the output library.",
#  "_key": """Key object used to determine the "class" of the library implementation.
#               The equals method is used to determine equality.""",
# User defined fields may be anything. Depsets sare be combined when merging libraries.
# buildifier: disable=provider-params
ExtraLinkTimeLibraryInfo = provider("ExtraLinkTimeLibraryInfo")

_KeyInfo = provider("_KeyInfo", fields = ["build_library_func", "constant_fields", "depset_fields"])

def create_extra_link_time_library(*, build_library_func, **kwargs):
    cc_internal.check_toplevel(build_library_func)
    return ExtraLinkTimeLibraryInfo(
        build_library_func = build_library_func,
        # Key to identify the "class" of a StarlarkDefinedLinkTimeLibrary. Uses the build function and
        # the split between depset and non-depset parameters to determine equality.
        _key = _KeyInfo(
            build_library_func = build_library_func,
            constant_fields = [k for k, v in kwargs.items() if type(v) != "depset"],
            depset_fields = [k for k, v in kwargs.items() if type(v) == "depset"],
        ),
        **kwargs
    )
