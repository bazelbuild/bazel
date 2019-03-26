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

load(":utils.bzl", "expand_pyversion_template")

def _py_runtime_pair_impl(ctx):
    if ctx.attr.py2_runtime != None:
        py2_runtime = ctx.attr.py2_runtime[PyRuntimeInfo]
        if py2_runtime.python_version != "PY2":
            fail("The Python runtime in the 'py2_runtime' attribute did not have " +
                 "version 'PY2'")
    else:
        py2_runtime = None

    if ctx.attr.py3_runtime != None:
        py3_runtime = ctx.attr.py3_runtime[PyRuntimeInfo]
        if py3_runtime.python_version != "PY3":
            fail("The Python runtime in the 'py3_runtime' attribute did not have " +
                 "version 'PY3'")
    else:
        py3_runtime = None

    return [platform_common.ToolchainInfo(
        py2_runtime = py2_runtime,
        py3_runtime = py3_runtime,
    )]

py_runtime_pair = rule(
    implementation = _py_runtime_pair_impl,
    attrs = {
        "py2_runtime": attr.label(providers = [PyRuntimeInfo], doc = """\
The runtime to use for Python 2 targets. Must have `python_version` set to
`PY2`.
"""),
        "py3_runtime": attr.label(providers = [PyRuntimeInfo], doc = """\
The runtime to use for Python 3 targets. Must have `python_version` set to
`PY3`.
"""),
    },
    doc = """\
A toolchain rule for Python.

This wraps up to two Python runtimes, one for Python 2 and one for Python 3.
The rule consuming this toolchain will choose which runtime is appropriate.
Either runtime may be omitted, in which case the resulting toolchain will be
unusable for building Python code using that version.

Usually the wrapped runtimes are declared using the `py_runtime` rule, but any
rule returning a `PyRuntimeInfo` provider may be used.

This rule returns a `platform_common.ToolchainInfo` provider with the following
schema:

```python
platform_common.ToolchainInfo(
    py2_runtime = <PyRuntimeInfo or None>,
    py3_runtime = <PyRuntimeInfo or None>,
)
```

Example usage:

```python
# In your BUILD file...

load("@bazel_tools//tools/python/toolchain.bzl", "py_runtime_pair")

py_runtime(
    name = "my_py2_runtime",
    interpreter_path = "/system/python2",
    python_version = "PY2",
)

py_runtime(
    name = "my_py3_runtime",
    interpreter_path = "/system/python3",
    python_version = "PY3",
)

py_runtime_pair(
    name = "my_py_runtime_pair",
    py2_runtime = ":my_py2_runtime",
    py3_runtime = ":my_py3_runtime",
)

toolchain(
    name = "my_toolchain",
    target_compatible_with = <...>,
    toolchain = ":my_py_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)
```

```python
# In your WORKSPACE...

register_toolchains("//my_pkg:my_toolchain")
```
""",
)

# TODO(#7844): Add support for a windows (.bat) version of the autodetecting
# toolchain, based on the "py" wrapper (e.g. "py -2" and "py -3"). Use select()
# in the template attr of the _generate_*wrapper targets.

def define_autodetecting_toolchain(name, pywrapper_template):
    """Defines the autodetecting Python toolchain.

    For use only by @bazel_tools//tools/python:BUILD; see the documentation
    comment there.

    Args:
        name: The name of the toolchain to introduce. Must have value
            "autodetecting_toolchain". This param is present only to make the
            BUILD file more readable.
        pywrapper_template: The label of the pywrapper_template.txt file.
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
        visibility = ["//visibility:private"],
    )

    # Note that the pywrapper script is a .sh file, not a sh_binary target. If
    # we needed to make it a proper shell target, e.g. because it needed to
    # access runfiles and needed to depend on the runfiles library, then we'd
    # have to use a workaround to allow it to be depended on by py_runtime. See
    # https://github.com/bazelbuild/bazel/issues/4286#issuecomment-475661317.

    native.py_runtime(
        name = "_autodetecting_py2_runtime",
        interpreter = ":py2wrapper.sh",
        python_version = "PY2",
        visibility = ["//visibility:private"],
    )

    native.py_runtime(
        name = "_autodetecting_py3_runtime",
        interpreter = ":py3wrapper.sh",
        python_version = "PY3",
        visibility = ["//visibility:private"],
    )

    py_runtime_pair(
        name = "_autodetecting_py_runtime_pair",
        py2_runtime = ":_autodetecting_py2_runtime",
        py3_runtime = ":_autodetecting_py3_runtime",
        visibility = ["//visibility:public"],
    )

    native.toolchain(
        name = name,
        toolchain = ":_autodetecting_py_runtime_pair",
        toolchain_type = ":toolchain_type",
        visibility = ["//visibility:public"],
    )
