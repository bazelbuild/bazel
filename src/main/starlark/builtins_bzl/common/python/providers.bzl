# Copyright 2022 The Bazel Authors. All rights reserved.
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
"""Providers for Python rules."""

DEFAULT_STUB_SHEBANG = "#!/usr/bin/env python3"
_PYTHON_VERSION_VALUES = ["PY2", "PY3"]

def _PyRuntimeInfo_init(
        *,
        interpreter_path = None,
        interpreter = None,
        files = None,
        coverage_tool = None,
        coverage_files = None,
        python_version,
        stub_shebang = None):
    if (interpreter_path == None) == (interpreter == None):
        fail("exactly one of interpreter_path or interpreter must be set")
    if (interpreter == None) != (files == None):
        fail("interpreter and files must both be set or neither must be set")
    if (coverage_tool == None) == (coverage_files == None):
        fail("coverage_tool and coverage_files must both be set or neither must be set")
    if python_version not in _PYTHON_VERSION_VALUES:
        fail("invalid python_version: '{}'; must be one of {}".format(
            python_version,
            _PYTHON_VERSION_VALUES,
        ))
    if not stub_shebang:
        stub_shebang = DEFAULT_STUB_SHEBANG
    return {
        "interpreter_path": interpreter_path,
        "interpreter": interpreter,
        "files": files,
        "coverage_tool": coverage_tool,
        "coverage_files": coverage_files,
        "python_version": python_version,
        "stub_shebang": stub_shebang,
    }

PyRuntimeInfo, _unused_raw_py_runtime_info_ctor = provider(
    doc = """Contains information about a Python runtime, as returned by the `py_runtime`
rule.

A Python runtime describes either a *platform runtime* or an *in-build runtime*.
A platform runtime accesses a system-installed interpreter at a known path,
whereas an in-build runtime points to a `File` that acts as the interpreter. In
both cases, an "interpreter" is really any executable binary or wrapper script
that is capable of running a Python script passed on the command line, following
the same conventions as the standard CPython interpreter.
""",
    init = _PyRuntimeInfo_init,
    fields = {
        "interpreter_path": (
            "If this is a platform runtime, this field is the absolute " +
            "filesystem path to the interpreter on the target platform. " +
            "Otherwise, this is `None`."
        ),
        "interpreter": (
            "If this is an in-build runtime, this field is a `File` representing " +
            "the interpreter. Otherwise, this is `None`. Note that an in-build " +
            "runtime can use either a prebuilt, checked-in interpreter or an " +
            "interpreter built from source."
        ),
        "files": (
            "If this is an in-build runtime, this field is a `depset` of `File`s" +
            "that need to be added to the runfiles of an executable target that " +
            "uses this runtime (in particular, files needed by `interpreter`). " +
            "The value of `interpreter` need not be included in this field. If " +
            "this is a platform runtime then this field is `None`."
        ),
        "coverage_tool": (
            "If set, this field is a `File` representing tool used for collecting code coverage information from python tests. Otherwise, this is `None`."
        ),
        "coverage_files": (
            "The files required at runtime for using `coverage_tool`. " +
            "Will be `None` if no `coverage_tool` was provided."
        ),
        "python_version": (
            "Indicates whether this runtime uses Python major version 2 or 3. " +
            "Valid values are (only) `\"PY2\"` and " +
            "`\"PY3\"`."
        ),
        "stub_shebang": (
            "\"Shebang\" expression prepended to the bootstrapping Python stub " +
            "script used when executing `py_binary` targets.  Does not " +
            "apply to Windows."
        ),
    },
)

PyInfo = _builtins.toplevel.PyInfo
