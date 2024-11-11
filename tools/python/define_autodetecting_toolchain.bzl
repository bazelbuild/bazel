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
"""Definitions related to the Python toolchain."""

load("@rules_python//python:py_runtime.bzl", "py_runtime")
load("@rules_python//python:py_runtime_pair.bzl", "py_runtime_pair")
load(":utils.bzl", "expand_pyversion_template")

# TODO(#7844): Add support for a windows (.bat) version of the autodetecting
# toolchain, based on the "py" wrapper (e.g. "py -2" and "py -3"). Use select()
# in the template attr of the _generate_*wrapper targets.

def define_autodetecting_toolchain(
        name,
        pywrapper_template,
        windows_config_setting):
    """Defines the autodetecting Python toolchain.

    This includes both strict and non-strict variants.

    For use only by @bazel_tools//tools/python:BUILD; see the documentation
    comment there.

    Args:
        name: The name of the toolchain to introduce. Must have value
            "autodetecting_toolchain". This param is present only to make the
            BUILD file more readable.
        pywrapper_template: The label of the pywrapper_template.txt file.
        windows_config_setting: The label of a config_setting that matches when
            the platform is windows, in which case the toolchain is configured
            in a way that triggers a workaround for #7844.
    """
    if native.package_name() != "tools/python":
        fail("define_autodetecting_toolchain() is private to " +
             "@bazel_tools//tools/python")
    if name != "autodetecting_toolchain":
        fail("Python autodetecting toolchain must be named " +
             "'autodetecting_toolchain'")

    expand_pyversion_template(
        name = "_generate_wrappers",
        template = pywrapper_template,
        out2 = ":py2wrapper.sh",
        out3 = ":py3wrapper.sh",
        out2_nonstrict = ":py2wrapper_nonstrict.sh",
        out3_nonstrict = ":py3wrapper_nonstrict.sh",
        visibility = ["//visibility:private"],
    )

    # Note that the pywrapper script is a .sh file, not a sh_binary target. If
    # we needed to make it a proper shell target, e.g. because it needed to
    # access runfiles and needed to depend on the runfiles library, then we'd
    # have to use a workaround to allow it to be depended on by py_runtime. See
    # https://github.com/bazelbuild/bazel/issues/4286#issuecomment-475661317.

    py_runtime(
        name = "_autodetecting_py3_runtime",
        interpreter = ":py3wrapper.sh",
        python_version = "PY3",
        stub_shebang = "#!/usr/bin/env python3",
        visibility = ["//visibility:private"],
    )

    py_runtime(
        name = "_autodetecting_py3_runtime_nonstrict",
        interpreter = ":py3wrapper_nonstrict.sh",
        python_version = "PY3",
        stub_shebang = "#!/usr/bin/env python3",
        visibility = ["//visibility:private"],
    )

    # This is a dummy runtime whose interpreter_path triggers the native rule
    # logic to use the legacy behavior on Windows.
    # TODO(#7844): Remove this target.
    py_runtime(
        name = "_magic_sentinel_runtime",
        interpreter_path = "/_magic_pyruntime_sentinel_do_not_use",
        python_version = "PY3",
        visibility = ["//visibility:private"],
    )

    py_runtime_pair(
        name = "_autodetecting_py_runtime_pair",
        py3_runtime = select({
            # If we're on windows, inject the sentinel to tell native rule logic
            # that we attempted to use the autodetecting toolchain and need to
            # switch back to legacy behavior.
            # TODO(#7844): Remove this hack.
            windows_config_setting: ":_magic_sentinel_runtime",
            "//conditions:default": ":_autodetecting_py3_runtime",
        }),
        visibility = ["//visibility:public"],
    )

    py_runtime_pair(
        name = "_autodetecting_py_runtime_pair_nonstrict",
        py3_runtime = select({
            # Same hack as above.
            # TODO(#7844): Remove this hack.
            windows_config_setting: ":_magic_sentinel_runtime",
            "//conditions:default": ":_autodetecting_py3_runtime_nonstrict",
        }),
        visibility = ["//visibility:public"],
    )

    native.toolchain(
        name = name,
        toolchain = ":_autodetecting_py_runtime_pair",
        toolchain_type = ":toolchain_type",
        visibility = ["//visibility:public"],
    )

    native.toolchain(
        name = name + "_nonstrict",
        toolchain = ":_autodetecting_py_runtime_pair_nonstrict",
        toolchain_type = ":toolchain_type",
        visibility = ["//visibility:public"],
    )
