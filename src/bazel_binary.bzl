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

"""Rule for building the Bazel binary."""

def _bazel_binary_impl(ctx):
    client = ctx.file.client
    workspace_zip = ctx.file.workspace_zip
    out = ctx.outputs.out

    # Intermediate file for concatenation.
    # This prevents modifying the final output in-place across multiple actions.
    concat_out = ctx.actions.declare_file(ctx.label.name + ".concat")

    # Step 1: Concatenate client binary and workspace zip.
    ctx.actions.run_shell(
        inputs = [client, workspace_zip],
        outputs = [concat_out],
        command = "cat {client} {workspace_zip} > {out}".format(
            client = client.path,
            workspace_zip = workspace_zip.path,
            out = concat_out.path,
        ),
        mnemonic = "BazelConcat",
        progress_message = "Concatenating client and zip for %s" % ctx.label.name,
    )

    # Step 2: Copy to final destination, adjust zip offsets, and make executable.
    # We must copy first because zip -qA modifies the file in-place.
    ctx.actions.run_shell(
        inputs = [concat_out],
        outputs = [out],
        command = "cp {input} {out} && zip -qA {out} && chmod a+x {out}".format(
            input = concat_out.path,
            out = out.path,
        ),
        use_default_shell_env = True,
        mnemonic = "BazelZipAdjust",
        progress_message = "Adjusting zip offsets for %s" % ctx.label.name,
    )

    return [DefaultInfo(
        executable = out,
        files = depset([out]),
    )]

bazel_binary = rule(
    implementation = _bazel_binary_impl,
    attrs = {
        "client": attr.label(allow_single_file = True, mandatory = True),
        "workspace_zip": attr.label(allow_single_file = True, mandatory = True),
        "out": attr.output(mandatory = True),
    },
    executable = True,
)
