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

# Keep in sync with default value of --experimental_better_python_version_mixing.
_DEFAULT_EXPERIMENTAL_FLAG = False

# Keep in sync with PythonVersion.DEFAULT_TARGET_VALUE.
_DEFAULT_PYTHON_VERSION = "PY2"

def _python_version_flag_impl(ctx):
    # Convert the string-encoded tri-value to a boolean, where unset yields the
    # default value.
    experimental = {
        _TRUE: True,
        _FALSE: False,
        _UNSET: _DEFAULT_EXPERIMENTAL_FLAG,
    }[ctx.attr.experimental_flag]

    # Version is determined using the same logic as in PythonOptions#getPythonVersion:
    #   1. Under the experimental flag, consult --python_version first,
    #      otherwise ignore it.
    #   2. Next fall back on --force_python.
    #   3. Final fallback is on the hardcoded default.
    if experimental and ctx.attr.python_version_flag != _UNSET:
        version = ctx.attr.python_version_flag
    elif ctx.attr.force_python_flag != _UNSET:
        version = ctx.attr.force_python_flag
    else:
        version = _DEFAULT_PYTHON_VERSION

    if version not in ["PY2", "PY3"]:
        fail("Internal error: _python_version_flag should only be able to " +
             "match 'PY2' or 'PY3'")
    return [config_common.FeatureFlagInfo(value = version)]

_python_version_flag = rule(
    implementation = _python_version_flag_impl,
    attrs = {
        "experimental_flag": attr.string(mandatory = True, values = [_TRUE, _FALSE, _UNSET]),
        "force_python_flag": attr.string(mandatory = True, values = [_PY2, _PY3, _UNSET]),
        "python_version_flag": attr.string(mandatory = True, values = [_PY2, _PY3, _UNSET]),
    },
)

def define_python_version_flag(name):
    """Defines the target to expose the Python version to select().

    For use only by @bazel_tools//python:BUILD; see the documentation comment
    there.

    Args:
        name: The name of the target to introduce.
    """

    # Config settings for the underlying native flags we depend on:
    # --experimental_better_python_version_mixing, --force_python, and
    # --python_version.
    native.config_setting(
        name = "_experimental_false",
        values = {"experimental_better_python_version_mixing": "false"},
        visibility = ["//visibility:private"],
    )
    native.config_setting(
        name = "_experimental_true",
        values = {"experimental_better_python_version_mixing": "true"},
        visibility = ["//visibility:private"],
    )
    native.config_setting(
        name = "_force_python_setting_PY2",
        values = {"force_python": "PY2"},
        visibility = ["//visibility:private"],
    )
    native.config_setting(
        name = "_force_python_setting_PY3",
        values = {"force_python": "PY3"},
        visibility = ["//visibility:private"],
    )
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

    _python_version_flag(
        name = name,
        experimental_flag = select({
            ":_experimental_false": _FALSE,
            ":_experimental_true": _TRUE,
            "//conditions:default": _UNSET,
        }),
        force_python_flag = select({
            ":_force_python_setting_PY2": _PY2,
            ":_force_python_setting_PY3": _PY3,
            "//conditions:default": _UNSET,
        }),
        python_version_flag = select({
            ":_python_version_setting_PY2": _PY2,
            ":_python_version_setting_PY3": _PY3,
            "//conditions:default": _UNSET,
        }),
        visibility = ["//visibility:public"],
    )
