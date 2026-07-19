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

"""Hermetic builtins archive rule."""

def _format_zip_entry(src):
    package_prefix = src.owner.package + "/"
    relative_path = src.short_path.partition(package_prefix)[2]
    return "builtins_bzl/{}={}".format(relative_path, src.path)

def _builtins_zip_impl(ctx):
    args = ctx.actions.args()
    args.add("cC")
    args.add(ctx.outputs.out)
    args.add_all(ctx.files.srcs, map_each = _format_zip_entry)

    ctx.actions.run(
        executable = ctx.executable.zipper,
        arguments = [args],
        inputs = ctx.files.srcs,
        outputs = [ctx.outputs.out],
        mnemonic = "BuiltinsZip",
        progress_message = "Building %{output}",
    )

builtins_zip = rule(
    implementation = _builtins_zip_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "out": attr.output(mandatory = True),
        "zipper": attr.label(
            executable = True,
            cfg = "exec",
        ),
    },
)
