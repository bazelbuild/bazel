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
Extra link-time library functionality.

CcInfo maintains a list of extra libraries to include in a link. These are non-C++ libraries that
are built from inputs gathered from all the dependencies. The dependencies have no way to
coordinate, so each one will add an ExtraLinkTimeLibrary to its CcLinkingContextInfo.
"""

load(":common/cc/cc_helper_internal.bzl", "check_private_api")

_cc_internal = _builtins.internal.cc_internal

# An implementation of ExtraLinkTimeLibrary that uses functions and data passed in from Starlark.

# Fields are intentionally not specified as they may be defined by users.
# Common fields are:
#  "build_library_func": "Starlark function to create the output library.",
#  "_key": """Key object used to determine the "class" of the library implementation.
#               The equals method is used to determine equality.""",
# User defined fields may be anything. Depsets are be combined when merging libraries.
# buildifier: disable=provider-params
ExtraLinkTimeLibraryInfo = provider("ExtraLinkTimeLibraryInfo")

_KeyInfo = provider("_KeyInfo", fields = ["build_library_func", "constant_fields", "depset_fields"])

def create_extra_link_time_library(*, build_library_func, **kwargs):
    """An extra library to include in a link. The actual library is built at link time.

    Exposed as cc_common.create_extra_link_time_library.

    This can be used for non-C++ inputs to a C++ link. A class that implements this interface will
    support transitively gathering all inputs from link dependencies, and then combine them all
    ogether into a set of C++ libraries.

    Any implementations must be immutable (and therefore thread-safe), because this is passed
    between rules and accessed in a multi-threaded context.

    Args:
      build_library_func: A function that takes a rule context, static_mode, for_dynamic_library,
        and kwargs, and returns a tuple of (linker_input, runtime_library): `tuple[LinkerInputInfo,
        File]`.
      **kwargs: Additional fields to pass to the build function.

    Returns:
      ExtraLinkTimeLibraryInfo.
    """
    _cc_internal.check_toplevel(build_library_func)
    return ExtraLinkTimeLibraryInfo(
        build_library_func = build_library_func,
        # Key to identify the "class" of a StarlarkDefinedLinkTimeLibrary. Uses the build function and
        # the split between depset and non-depset parameters to determine equality.
        _key = _KeyInfo(
            build_library_func = build_library_func,
            # _KeyInfo is used in a dict, so all of its fields must be frozen/hashable.
            constant_fields = _cc_internal.freeze([k for k, v in kwargs.items() if type(v) != "depset"]),
            depset_fields = _cc_internal.freeze([k for k, v in kwargs.items() if type(v) == "depset"]),
        ),
        **kwargs
    )

ExtraLinkTimeLibrariesInfo = provider(
    "ExtraLinkTimeLibrariesInfo",
    fields = {
        "libraries": "A list of (ExtraLinkTimeLibraryInfo) extra libraries.",
    },
)

_EMPTY = ExtraLinkTimeLibrariesInfo(
    libraries = [],
)

def create_extra_link_time_libraries(library):
    """Creates ExtraLinkTimeLibrariesInfo.

    Args:
      library: A single ExtraLinkTimeLibraryInfo (or None).

    Returns:
      ExtraLinkTimeLibrariesInfo.
    """
    if library == None:
        return _EMPTY
    libraries = _cc_internal.freeze([library])
    return ExtraLinkTimeLibrariesInfo(
        libraries = libraries,
    )

def _merge_values(values):
    if not values:
        return None
    first_value = values[0]
    if type(first_value) == "depset":
        return depset(transitive = values, order = "topological")
    else:
        # All the constant values should be the same,
        # but this wasn't always enforced and they aren't :(.
        return first_value

def merge_extra_link_time_libraries(libraries):
    """Merges a list of ExtraLinkTimeLibraryInfos.

    Args:
      libraries: A list of ExtraLinkTimeLibraryInfos.

    Returns:
      ExtraLinkTimeLibrariesInfo.
    """
    if not libraries:
        return _EMPTY

    merged_libraries = {}
    for extra_library in libraries:
        for library in extra_library.libraries:
            key = library._key
            if key not in merged_libraries:
                merged_libraries[key] = []
            merged_libraries[key].append(library)

    result = []
    for key, libs_to_merge in merged_libraries.items():
        if len(libs_to_merge) == 1:
            result.append(libs_to_merge[0])
            continue

        merged_fields = {"build_library_func": key.build_library_func, "_key": key}

        all_keys = key.constant_fields + key.depset_fields

        for field in all_keys:
            merged_fields[field] = _merge_values([getattr(lib, field) for lib in libs_to_merge])

        result.append(ExtraLinkTimeLibraryInfo(**merged_fields))
    return ExtraLinkTimeLibrariesInfo(
        libraries = _cc_internal.freeze(result),
    )

def build_libraries(extra_libraries, ctx, static_mode, for_dynamic_library):
    """Builds the extra link-time libraries.

    Args:
      extra_libraries: A list of ExtraLinkTimeLibraryInfo objects.
      ctx: The rule context.
      static_mode: Whether the link is static.
      for_dynamic_library: Whether the link is for a dynamic library.

    Returns:
      A tuple of (linker_inputs, runtime_libraries): tuple[depset[LinkerInputInfo], depset[File]].
    """
    check_private_api()
    transitive_linker_inputs = []
    transitive_runtime_libraries = []
    for library in extra_libraries:
        kwargs = {}
        for key in dir(library):
            if key not in ["build_library_func", "_key"]:
                kwargs[key] = getattr(library, key)
        (linker_input, runtime_library) = library.build_library_func(
            ctx,
            static_mode,
            for_dynamic_library,
            **kwargs
        )
        transitive_linker_inputs.append(linker_input)
        transitive_runtime_libraries.append(runtime_library)

    return (
        depset(transitive = transitive_linker_inputs),
        depset(transitive = transitive_runtime_libraries),
    )
