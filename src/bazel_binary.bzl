# Copyright 2026 The Bazel Authors. All rights reserved.
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

"""Custom rule for packaging Bazel binary."""

def _bazel_binary_impl(ctx):
    client = ctx.file.client
    package_zip = ctx.file.package_zip
    cat_binary = ctx.executable._cat_binary
    adjust_sfx = ctx.executable._adjust_sfx
    output = ctx.outputs.out

    unaligned = ctx.actions.declare_file(output.basename + ".unaligned")

    # Concatenate client and zip
    ctx.actions.run(
        inputs = [client, package_zip],
        outputs = [unaligned],
        executable = cat_binary,
        arguments = [client.path, package_zip.path, unaligned.path],
        mnemonic = "ConcatClientAndZip",
        progress_message = "Concatenating client and zip for %{ctx.label.name}",
    )

    # Adjust SFX
    ctx.actions.run(
        inputs = [unaligned],
        outputs = [output],
        executable = adjust_sfx,
        arguments = [unaligned.path, output.path],
        mnemonic = "AdjustSfx",
        progress_message = "Adjusting SFX for %{ctx.label.name}",
    )

    return [DefaultInfo(
        executable = output,
        files = depset([output]),
    )]

bazel_binary = rule(
    implementation = _bazel_binary_impl,
    attrs = {
        "client": attr.label(allow_single_file = True, mandatory = True),
        "package_zip": attr.label(allow_single_file = True, mandatory = True),
        "out": attr.output(mandatory = True),
        "_cat_binary": attr.label(
            default = Label("//src/tools/simple_catter:simple_catter"),
            executable = True,
            cfg = "exec",
        ),
        "_adjust_sfx": attr.label(
            default = Label("//src/java_tools/singlejar/java/com/google/devtools/build/zip:adjust_sfx"),
            executable = True,
            cfg = "exec",
        ),
    },
    executable = True,
)
