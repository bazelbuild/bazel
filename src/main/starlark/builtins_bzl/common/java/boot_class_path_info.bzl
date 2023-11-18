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
# limitations under the License.

"""
Definition of the BootClassPathInfo provider.
"""

load(":common/paths.bzl", "paths")

def _init(bootclasspath = [], auxiliary = [], system = None):
    """The <code>BootClassPathInfo</code> constructor.

    Args:
        bootclasspath: ([File])
        auxiliary: ([File])
        system: ([File]|File|None)
    """
    if not system:  # None or []
        system_inputs = depset()
        system_path = None
    elif type(system) == "File":
        system_inputs = depset([system])
        if not system.is_directory:
            fail("for system,", system, "is not a directory")
        system_path = system.path
    elif type(system) == type([]):
        system_inputs = depset(system)
        system_paths = [input.path for input in system if input.basename == "release"]
        if not system_paths:
            fail("for system, expected inputs to contain 'release'")
        system_path = paths.dirname(system_paths[0])
    else:
        fail("for system, got", type(system), ", want File, sequence, or None")

    return {
        "bootclasspath": depset(bootclasspath),
        "_auxiliary": depset(auxiliary),
        "_system_inputs": system_inputs,
        "_system_path": system_path,
    }

BootClassPathInfo, _new_bootclasspathinfo = provider(
    doc = "Information about the system APIs for a Java compilation.",
    fields = [
        "bootclasspath",
        # private
        "_auxiliary",
        "_system_inputs",
        "_system_path",
    ],
    init = _init,
)
