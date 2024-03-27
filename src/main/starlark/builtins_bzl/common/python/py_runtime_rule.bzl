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
"""Implementation of py_runtime rule."""

load(":common/paths.bzl", "paths")
load(":common/python/attributes.bzl", "NATIVE_RULES_ALLOWLIST_ATTRS")
load(":common/python/common.bzl", "check_native_allowed")
load(":common/python/providers.bzl", "DEFAULT_BOOTSTRAP_TEMPLATE", "DEFAULT_STUB_SHEBANG", _PyRuntimeInfo = "PyRuntimeInfo")

_py_builtins = _builtins.internal.py_builtins

def _py_runtime_impl(ctx):
    check_native_allowed(ctx)
    interpreter_path = ctx.attr.interpreter_path or None  # Convert empty string to None
    interpreter = ctx.file.interpreter
    if (interpreter_path and interpreter) or (not interpreter_path and not interpreter):
        fail("exactly one of the 'interpreter' or 'interpreter_path' attributes must be specified")

    runtime_files = depset(transitive = [
        t[DefaultInfo].files
        for t in ctx.attr.files
    ])

    hermetic = bool(interpreter)
    if not hermetic:
        if runtime_files:
            fail("if 'interpreter_path' is given then 'files' must be empty")
        if not paths.is_absolute(interpreter_path):
            fail("interpreter_path must be an absolute path")

    if ctx.attr.coverage_tool:
        coverage_di = ctx.attr.coverage_tool[DefaultInfo]

        if _py_builtins.is_singleton_depset(coverage_di.files):
            coverage_tool = coverage_di.files.to_list()[0]
        elif coverage_di.files_to_run and coverage_di.files_to_run.executable:
            coverage_tool = coverage_di.files_to_run.executable
        else:
            fail("coverage_tool must be an executable target or must produce exactly one file.")

        coverage_files = depset(transitive = [
            coverage_di.files,
            coverage_di.default_runfiles.files,
        ])
    else:
        coverage_tool = None
        coverage_files = None

    python_version = ctx.attr.python_version
    if python_version == "_INTERNAL_SENTINEL":
        if ctx.fragments.py.use_toolchains:
            fail(
                "When using Python toolchains, this attribute must be set explicitly to either 'PY2' " +
                "or 'PY3'. See https://github.com/bazelbuild/bazel/issues/7899 for more " +
                "information. You can temporarily avoid this error by reverting to the legacy " +
                "Python runtime mechanism (`--incompatible_use_python_toolchains=false`).",
            )
        else:
            python_version = ctx.fragments.py.default_python_version

    # TODO: Uncomment this after --incompatible_python_disable_py2 defaults to true
    # if ctx.fragments.py.disable_py2 and python_version == "PY2":
    #     fail("Using Python 2 is not supported and disabled; see " +
    #          "https://github.com/bazelbuild/bazel/issues/15684")

    return [
        _PyRuntimeInfo(
            interpreter_path = interpreter_path or None,
            interpreter = interpreter,
            files = runtime_files if hermetic else None,
            coverage_tool = coverage_tool,
            coverage_files = coverage_files,
            python_version = python_version,
            stub_shebang = ctx.attr.stub_shebang,
            bootstrap_template = ctx.file.bootstrap_template,
        ),
        DefaultInfo(
            files = runtime_files,
            runfiles = ctx.runfiles(),
        ),
    ]

# Bind to the name "py_runtime" to preserve the kind/rule_class it shows up
# as elsewhere.
py_runtime = rule(
    implementation = _py_runtime_impl,
    doc = """
Represents a Python runtime used to execute Python code.

A `py_runtime` target can represent either a *platform runtime* or an *in-build
runtime*. A platform runtime accesses a system-installed interpreter at a known
path, whereas an in-build runtime points to an executable target that acts as
the interpreter. In both cases, an "interpreter" means any executable binary or
wrapper script that is capable of running a Python script passed on the command
line, following the same conventions as the standard CPython interpreter.

A platform runtime is by its nature non-hermetic. It imposes a requirement on
the target platform to have an interpreter located at a specific path. An
in-build runtime may or may not be hermetic, depending on whether it points to
a checked-in interpreter or a wrapper script that accesses the system
interpreter.

# Example

```
py_runtime(
    name = "python-2.7.12",
    files = glob(["python-2.7.12/**"]),
    interpreter = "python-2.7.12/bin/python",
)

py_runtime(
    name = "python-3.6.0",
    interpreter_path = "/opt/pyenv/versions/3.6.0/bin/python",
)
```
""",
    fragments = ["py"],
    attrs = NATIVE_RULES_ALLOWLIST_ATTRS | {
        "files": attr.label_list(
            allow_files = True,
            doc = """
For an in-build runtime, this is the set of files comprising this runtime.
These files will be added to the runfiles of Python binaries that use this
runtime. For a platform runtime this attribute must not be set.
""",
        ),
        "interpreter": attr.label(
            allow_single_file = True,
            doc = """
For an in-build runtime, this is the target to invoke as the interpreter. For a
platform runtime this attribute must not be set.
""",
        ),
        "interpreter_path": attr.string(doc = """
For a platform runtime, this is the absolute path of a Python interpreter on
the target platform. For an in-build runtime this attribute must not be set.
"""),
        "coverage_tool": attr.label(
            allow_files = False,
            doc = """
This is a target to use for collecting code coverage information from `py_binary`
and `py_test` targets.

If set, the target must either produce a single file or be an executable target.
The path to the single file, or the executable if the target is executable,
determines the entry point for the python coverage tool.  The target and its
runfiles will be added to the runfiles when coverage is enabled.

The entry point for the tool must be loadable by a Python interpreter (e.g. a
`.py` or `.pyc` file).  It must accept the command line arguments
of coverage.py (https://coverage.readthedocs.io), at least including
the `run` and `lcov` subcommands.
""",
        ),
        "python_version": attr.string(
            default = "_INTERNAL_SENTINEL",
            values = ["PY2", "PY3", "_INTERNAL_SENTINEL"],
            doc = """
Whether this runtime is for Python major version 2 or 3. Valid values are `"PY2"`
and `"PY3"`.

The default value is controlled by the `--incompatible_py3_is_default` flag.
However, in the future this attribute will be mandatory and have no default
value.
            """,
        ),
        "stub_shebang": attr.string(
            default = DEFAULT_STUB_SHEBANG,
            doc = """
"Shebang" expression prepended to the bootstrapping Python stub script
used when executing `py_binary` targets.

See https://github.com/bazelbuild/bazel/issues/8685 for
motivation.

Does not apply to Windows.
""",
        ),
        "bootstrap_template": attr.label(
            allow_single_file = True,
            default = DEFAULT_BOOTSTRAP_TEMPLATE,
            doc = """
The bootstrap script template file to use. Should have %python_binary%,
%workspace_name%, %main%, and %imports%.

This template, after expansion, becomes the executable file used to start the
process, so it is responsible for initial bootstrapping actions such as finding
the Python interpreter, runfiles, and constructing an environment to run the
intended Python application.

While this attribute is currently optional, it will become required when the
Python rules are moved out of Bazel itself.

The exact variable names expanded is an unstable API and is subject to change.
The API will become more stable when the Python rules are moved out of Bazel
itself.

See @bazel_tools//tools/python:python_bootstrap_template.txt for more variables.
""",
        ),
    },
)
