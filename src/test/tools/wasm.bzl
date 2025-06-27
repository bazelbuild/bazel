# Copyright 2025 The Bazel Authors. All rights reserved.
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

"""
This file contains a rule to compile WAT files to WebAssembly.
"""

def _wat_binary(ctx):
    out = ctx.actions.declare_file(ctx.attr.name + ".wasm")
    ctx.actions.run(
        mnemonic = "Wat2Wasm",
        executable = ctx.executable._wat2wasm,
        inputs = [ctx.file.src],
        outputs = [out],
        arguments = [
            ctx.file.src.path,
            "-o",
            out.path,
        ],
    )
    return DefaultInfo(files = depset([out]))

wat_binary = rule(
    implementation = _wat_binary,
    attrs = {
        "src": attr.label(
            allow_single_file = [".wat"],
            mandatory = True,
        ),
        "_wat2wasm": attr.label(
            default = "@wabt//src/tools:wat2wasm",
            executable = True,
            cfg = "exec",
        ),
    },
)
