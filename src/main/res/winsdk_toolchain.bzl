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

"""Toolchain definition for Windows Resource Compiler toolchains.

Usage:

    load(
        ":winsdk_toolchain.bzl",
        "WINDOWS_RESOURCE_COMPILER_TOOLCHAIN_TYPE",
        "windows_resource_compiler_toolchain",
    )

    windows_resource_compiler_toolchain(
        name = "foo_rc_toolchain",
        rc_path = "...",  # label of Resource Compiler or its wrapper
    )

    toolchain(
        name = "foo_rc",
        exec_compatible_with = [
            # Add constraints here, if applicable.
        ],
        target_compatible_with = [
            # Add constraints here, if applicable.
        ],
        toolchain = ":foo_rc_toolchain",
        toolchain_type = WINDOWS_RESOURCE_COMPILER_TOOLCHAIN_TYPE,
    )
"""

WINDOWS_RESOURCE_COMPILER_TOOLCHAIN_TYPE = "@io_bazel//src/main/res:toolchain_type"

WindowsResourceCompilerInfo = provider(
    fields = ["rc_exe"],
    doc = "Toolchain info for the Resource Compiler on Windows",
)

def _impl(ctx):
    toolchain_info = platform_common.ToolchainInfo(
        win_rc_info = WindowsResourceCompilerInfo(
            rc_exe = ctx.executable.rc_exe,
        ),
    )
    return [toolchain_info]

windows_resource_compiler_toolchain = rule(
    doc = "Toolchain rule for the Resource Compiler on Windows",
    implementation = _impl,
    attrs = {
        "rc_exe": attr.label(
            allow_files = True,
            executable = True,
            cfg = "host",
            doc = "Label of the resource compiler (or a wrapper script)",
        ),
    },
)
