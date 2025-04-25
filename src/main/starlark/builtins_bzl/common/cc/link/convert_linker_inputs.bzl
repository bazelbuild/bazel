# Copyright 2024 The Bazel Authors. All rights reserved.
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
"""Helper functions converting LibraryToLink-s to LegacyLinkerInput-s."""

cc_internal = _builtins.internal.cc_internal

def convert_library_to_link_list_to_linker_input_list(libraries_to_link, static_mode, for_dynamic_library, supports_dynamic_linker):
    """
    Converts LibraryToLink-s from CcLinkingContext-s to LegacyLinkerInput-s.

    This code is used for creating/linking a dynamic library or an executable, where all transitive
    libraries are input to the linker. Linking mode can be either static or dynamic.

    First the function flattens transitive CcLinkingContext-s in a topological/linking orders, then
    it flattens all LibraryToLink-s in them.

    For each LibraryToLink it creates one of the best suited LegacyLinkerInputs (static pic/nopic,
    interface, dynamic), prioritising the flavour based on the given parameters.

    Further on, LegacyLinkerInputs are converted to LibraryToLinkValues and passed as link build
    variables to generated the command line.

    Args:
      libraries_to_link: (list[LibraryToLink]) Libraries from dependencies.
      static_mode: (bool) True for `static`, False for `dynamic` linking mode.
      for_dynamic_library: (bool) True when creating a library. False for executable.
      supports_dynamic_linker: (bool) True when C++ toolchain supports_dynamic_linker. That is,
        toolchain can produce binaries that load shared libraries at runtime.
    Returns:
      (list[LibraryInput]) The selected libraries. LibraryInputs are subclass of LegacyLinkerInputs.
      The contain the information about libraries main file as well as it's object files. Only one
      of those is used in the link.
    """
    # TODO(b/338618120): This whole function shouldn't really exist. Eventually the processing
    # should go directly from LibraryToLink into LibraryToLinkValue. Most likely the conversion will
    # need to move into LibrariesToLinkCollector to achieve that.
    #
    # Some smaller goals: Reorder the if statements, to follow 4 possible distinct returned values.
    # Rewrite LegacyLinkerInputs to Starlark (no blockers). This changes need to be done with
    # performance in mind. Current implementation might already need improvements before the
    # Starlark cc_common.link may be turned on, because it's converting between Java and Starlark
    # types a lot and that might cause more garbage. But the garbage also can't be completely
    # removed until both LibraryToLink and LibraryToLinkValues are in Starlark.

    libraries = cc_internal.convert_library_to_link_list_to_linker_input_list(libraries_to_link, static_mode, for_dynamic_library, supports_dynamic_linker)
    return depset(libraries, order = "topological").to_list()  # filter duplicates
