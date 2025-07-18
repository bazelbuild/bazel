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
This module contains functionality for creating linker inputs for C++ rules.
"""

cc_internal = _builtins.internal.cc_internal

_LinkerInputInfo = provider(
    "LinkerInputInfo",
    fields = {
        "owner": "The owner of the linker input.",
        "libraries": "A depset of libraries to link.",
        "user_link_flags": "A list of user link flags.",
        "additional_inputs": "A depset of non-code inputs.",
        "linkstamps": "A depset of linkstamps.",
    },
)

def create_linker_input(
        *,
        owner,
        libraries = depset(),
        user_link_flags = [],
        additional_inputs = depset(),
        linkstamps = depset()):
    """Creates a LinkerInputInfo provider.

    Args:
        owner: The owner of the linker input.
        libraries: A depset of libraries to link.
        user_link_flags: A list of user link flags.
        additional_inputs: A depset of non-code inputs.
        linkstamps: A depset of linkstamps.

    Returns:
        A LinkerInputInfo provider.
    """
    options = []

    if type(user_link_flags) == "depset":
        options.extend(user_link_flags.to_list())
    elif type(user_link_flags) == "list":
        for flag in user_link_flags:
            if type(flag) == "string":
                options.append(flag)
            elif type(flag) == "list":
                options.extend(flag)
            else:
                fail("Elements of list in user_link_flags must be either Strings or lists.")

    return _LinkerInputInfo(
        owner = owner,
        libraries = cc_internal.freeze(libraries.to_list()),
        user_link_flags = cc_internal.freeze(options),
        additional_inputs = cc_internal.freeze(additional_inputs.to_list()),
        linkstamps = cc_internal.freeze(linkstamps.to_list()),
    )
