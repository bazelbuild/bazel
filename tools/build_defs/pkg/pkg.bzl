# Copyright 2015 The Bazel Authors. All rights reserved.
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
"""Rules for manipulation of various packaging."""

load(":path.bzl", "compute_data_path", "dest_path")

def _quote(filename, protect = "="):
    """Quote the filename, by escaping = by \\= and \\ by \\\\"""
    return filename.replace("\\", "\\\\").replace(protect, "\\" + protect)

def _pkg_tar_impl(ctx):
    """Implementation of the pkg_tar rule."""

    # Compute the relative path
    data_path = compute_data_path(ctx.outputs.out, ".")

    # Start building the arguments.
    args = ctx.actions.args()
    args.add("--output", ctx.outputs.out.path)
    args.add("--directory", ctx.attr.package_dir)

    file_inputs = ctx.files.srcs[:]
    for f in file_inputs:
        args.add("--file", "%s=%s" % (_quote(f.path), dest_path(f, data_path)))
    args.set_param_file_format("flag_per_line")
    args.use_param_file("@%s", use_always = False)

    ctx.actions.run(
        inputs = file_inputs,
        executable = ctx.executable.build_tar,
        arguments = [args],
        outputs = [ctx.outputs.out],
        mnemonic = "PackageTar",
        use_default_shell_env = True,
    )

# A rule for creating a tar file, see README.md
_real_pkg_tar = rule(
    implementation = _pkg_tar_impl,
    attrs = {
        "package_dir": attr.string(default = "/"),
        "srcs": attr.label_list(allow_files = True),
        "out": attr.output(),
        # Implicit dependencies.
        "build_tar": attr.label(
            default = Label("//tools/build_defs/pkg:build_tar"),
            cfg = "exec",
            executable = True,
            allow_files = True,
        ),
    },
)

def pkg_tar(name, **kwargs):
    _real_pkg_tar(
        name = name,
        out = name + ".tar",
        **kwargs
    )
