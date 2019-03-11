# Copyright 2019 The Bazel Authors. All rights reserved.
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
Finds the Java toolchain.

Returns the toolchain if enabled, and falls back to a toolchain constructed from
legacy toolchain selection.
"""

def find_java_toolchain(ctx, target):
    """
    Finds the Java toolchain.

    If the Java toolchain is in use, returns it.  Otherwise, returns a Java
    toolchain derived from legacy toolchain selection.

    Args:
      ctx: The rule context for which to find a toolchain.
      target: A java_toolchain target (for legacy toolchain resolution).

    Returns:
      A JavaToolchainInfo.
    """

    _ignore = [ctx]

    return target[java_common.JavaToolchainInfo]

def find_java_runtime_toolchain(ctx, target):
    """
    Finds the Java runtime.

    If the Java toolchain is in use, returns it.  Otherwise, returns a Java
    runtime derived from legacy toolchain selection.

    Args:
      ctx: The rule context for which to find a toolchain.
      target: A java_runtime target (for legacy toolchain resolution).

    Returns:
      A JavaRuntimeInfo.
    """

    _ignore = [ctx]

    return target[java_common.JavaRuntimeInfo]
