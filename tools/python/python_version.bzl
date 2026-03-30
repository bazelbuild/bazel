# Copyright 2018 The Bazel Authors. All rights reserved.
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

"""Utilities for selecting the Python major version (Python 2 vs Python 3)."""

_UNSET = "UNSET"
_FALSE = "FALSE"
_TRUE = "TRUE"
_PY2 = "PY2"
_PY3 = "PY3"

def _python_version_flag_impl(ctx):
    # Always return PY3 as PY2 is no longer supported.
    # TODO: Remove this file once all references to //tools/python:python_version are removed.
    return [config_common.FeatureFlagInfo(value = _PY3)]

_python_version_flag = rule(
    implementation = _python_version_flag_impl,
)

def define_python_version_flag(name):
    """Defines the target to expose the Python version to select().

    For use only by @bazel_tools//tools/python:BUILD; see the documentation
    comment there.

    Args:
        name: The name of the target to introduce. Must have value
            "python_version". This param is present only to make the BUILD file
            more readable.
    """
    if native.package_name() != "tools/python":
        fail("define_python_version_flag() is private to " +
             "@bazel_tools//tools/python")
    if name != "python_version":
        fail("Python version flag must be named 'python_version'")

    _python_version_flag(
        name = name,
        visibility = ["//visibility:public"],
    )
