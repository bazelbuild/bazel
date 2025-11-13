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

"""Starlark rule to compile RC files on Windows."""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(":winsdk_toolchain.bzl", "WINDOWS_RESOURCE_COMPILER_TOOLCHAIN_TYPE")

def _replace_ext(n, e):
    i = n.rfind(".")
    if i > 0:
        return n[:i] + e
    else:
        return n + e

def _compile_rc(ctx, rc_exe, rc_file, extra_inputs):
    """Compiles a single RC file to RES."""
    out = ctx.actions.declare_file(_replace_ext(rc_file.basename, ".res"))
    ctx.actions.run(
        inputs = [rc_file] + extra_inputs,
        outputs = [out],
        executable = rc_exe,
        arguments = ["/nologo", "/fo%s" % out.path, rc_file.path],
        mnemonic = "WindowsRc",
    )
    return out

def _windows_resources_impl(ctx):
    rc_toolchain = ctx.toolchains[WINDOWS_RESOURCE_COMPILER_TOOLCHAIN_TYPE].win_rc_info
    if not rc_toolchain.rc_exe:
        return [CcInfo()]

    compiled_resources = [
        _compile_rc(ctx, rc_toolchain.rc_exe, rc_file, ctx.files.resources)
        for rc_file in ctx.files.rc_files
    ]
    link_flags = [res.path for res in compiled_resources]
    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        additional_inputs = depset(compiled_resources),
        user_link_flags = depset(link_flags),
    )
    linking_context = cc_common.create_linking_context(
        linker_inputs = depset([linker_input]),
    )
    return [
        DefaultInfo(files = depset(compiled_resources)),
        CcInfo(linking_context = linking_context),
    ]

windows_resources = rule(
    implementation = _windows_resources_impl,
    attrs = {
        "rc_files": attr.label_list(
            mandatory = True,
            allow_files = [".rc"],
            doc = "Resource files to compile. Each file must have a different basename to avoid conflicting output files.",
        ),
        "resources": attr.label_list(
            allow_files = True,
            doc = "Additional input files that RC files reference, if any.",
        ),
    },
    fragments = ["cpp"],
    toolchains = [WINDOWS_RESOURCE_COMPILER_TOOLCHAIN_TYPE],
    doc = """Compiles Windows resources (RC files) to be embedded in the final cc_binary.

Accepts .rc files (with accompanying resources) and embeds them into the
cc_binary that depends on this target.

Example usage:

    windows_resources(
        name = "hello_resources",
        rc_files = [
            "hello.rc",
            "version.rc",
        ],
        resources = [
            "version.txt",
            "//images:app.ico",
        ],
    )

    cc_binary(
        name = "hello",
        srcs = ["main.cc"],
        deps = [":hello_resources"],
    )""",
)
