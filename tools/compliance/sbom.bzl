# Copyright 2023 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate an SBOM for a target."""

load(":gather_packages.bzl", "packages_used")

def _sbom_impl(ctx):
    # Gather all licenses and write information to one place

    # Now turn the big blob of data into something consumable.
    outputs = [ctx.outputs.out]
    args = ctx.actions.args()
    inps = ctx.files.packages_used
    if len(inps) != 1:
        fail("Got too many inputs for packages_used")
    inputs = [inps[0]]
    args.add("--packages_used", inps[0].path)
    args.add("--out", ctx.outputs.out.path)
    ctx.actions.run(
        mnemonic = "CreateSBOM",
        progress_message = "Creating SBOM for %s" % ctx.label,
        inputs = inputs,
        outputs = outputs,
        executable = ctx.executable._sbom_generator,
        arguments = [args],
    )
    return [DefaultInfo(files = depset(outputs))]

_sbom = rule(
    implementation = _sbom_impl,
    attrs = {
        "packages_used": attr.label(
            allow_files = True,
            mandatory = True,
        ),
        "out": attr.output(mandatory = True),
        "_sbom_generator": attr.label(
            default = Label("//tools/compliance:write_sbom_private"),
            executable = True,
            allow_files = True,
            cfg = "exec",
        ),
    },
)

def sbom(name, target, out = None):
    """Wrapper for sbom rule.

    Args:
        name: name
        target: Target to create sbom for
        out: output file name
    """
    packages = "_packages_" + name
    packages_used(
        name = packages,
        target = target,
        out = packages + ".json",
    )
    if not out:
        out = name + "_sbom.json"
    _sbom(
        name = name,
        out = out,
        packages_used = ":" + packages + ".json",
    )
