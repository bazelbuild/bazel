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
"""Define a toolchain rule for the shell."""

def _sh_toolchain_impl(ctx):
    """sh_toolchain rule implementation."""
    return [
        platform_common.ToolchainInfo(
            path = ctx.attr.path,
            launcher = ctx.executable.launcher,
            launcher_maker = ctx.executable.launcher_maker,
        ),
    ]

sh_toolchain = rule(
    doc = "A runtime toolchain for shell targets.",
    attrs = {
        "path": attr.string(
            doc = "Absolute path to the shell interpreter.",
            mandatory = True,
        ),
        "launcher": attr.label(
            doc = "The generic launcher binary to use to run sh_binary/sh_test targets (only used when targeting Windows).",
            cfg = "target",
            allow_single_file = True,
            executable = True,
        ),
        "launcher_maker": attr.label(
            doc = "The tool to use to create a target-specific launcher from the generic launcher binary (only used when targeting Windows).",
            cfg = "exec",
            allow_single_file = True,
            executable = True,
        ),
    },
    implementation = _sh_toolchain_impl,
)
