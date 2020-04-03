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
    # Version is determined using the same logic as in
    # PythonOptions#getPythonVersion:
    #
    #   1. Consult --python_version first, if present.
    #   2. Fallback on the default, which is governed by an incompatible change
    #      flag.
    if ctx.attr.python_version_flag != _UNSET:
        version = ctx.attr.python_version_flag
    else:
        version = _PY3 if ctx.attr.incompatible_py3_is_default_flag else _PY2

    if version not in ["PY2", "PY3"]:
        fail("Internal error: _python_version_flag should only be able to " +
             "match 'PY2' or 'PY3'")
    return [config_common.FeatureFlagInfo(value = version)]

_python_version_flag = rule(
    implementation = _python_version_flag_impl,
    attrs = {
        "python_version_flag": attr.string(mandatory = True, values = [_PY2, _PY3, _UNSET]),
        "incompatible_py3_is_default_flag": attr.bool(mandatory = True),
    },
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

    # Config settings for the underlying native flags we depend on:
    # --python_version and --incompatible_py3_is_default.
    native.config_setting(
        name = "_python_version_setting_PY2",
        values = {"python_version": "PY2"},
        visibility = ["//visibility:private"],
    )
    native.config_setting(
        name = "_python_version_setting_PY3",
        values = {"python_version": "PY3"},
        visibility = ["//visibility:private"],
    )
    native.config_setting(
        name = "_incompatible_py3_is_default_setting_false",
        values = {"incompatible_py3_is_default": "false"},
        visibility = ["//visibility:private"],
    )
    native.config_setting(
        name = "_incompatible_py3_is_default_setting_true",
        values = {"incompatible_py3_is_default": "true"},
        visibility = ["//visibility:private"],
    )

    _python_version_flag(
        name = name,
        python_version_flag = select({
            ":_python_version_setting_PY2": _PY2,
            ":_python_version_setting_PY3": _PY3,
            "//conditions:default": _UNSET,
        }),
        incompatible_py3_is_default_flag = select({
            ":_incompatible_py3_is_default_setting_false": False,
            ":_incompatible_py3_is_default_setting_true": True,
        }),
        visibility = ["//visibility:public"],
    )
