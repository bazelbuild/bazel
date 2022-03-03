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
"""mini_tar: A limited functionality tar utility."""

load(":path.bzl", "compute_data_path", "dest_path")

# Filetype to restrict inputs
tar_filetype = [".tar", ".tar.gz", ".tgz", ".tar.bz2"]

def _quote(filename, protect = "="):
    """Quote the filename, by escaping = by \\= and \\ by \\\\"""
    return filename.replace("\\", "\\\\").replace(protect, "\\" + protect)

def _mini_tar_impl(ctx):
    """Implementation of the mini_tar rule."""

    # Compute the relative path
    data_path = compute_data_path(ctx.outputs.out, ctx.attr.strip_prefix)

    # Start building the arguments.
    args = [
        "--output=" + ctx.outputs.out.path,
        "--directory=" + ctx.attr.package_dir,
        "--mode=" + ctx.attr.mode,
        "--owner=" + ctx.attr.owner,
    ]
    if ctx.attr.mtime != -1:  # Note: Must match default in rule def.
        args.append("--mtime=%d" % ctx.attr.mtime)

    file_inputs = ctx.files.srcs[:]
    args += [
        "--file=%s=%s" % (_quote(f.path), dest_path(f, data_path))
        for f in file_inputs
    ]
    arg_file = ctx.actions.declare_file(ctx.label.name + ".args")
    ctx.actions.write(arg_file, "\n".join(args))
    ctx.actions.run(
        inputs = file_inputs + ctx.files.deps + [arg_file],
        executable = ctx.executable.build_tar,
        arguments = ["--flagfile", arg_file.path],
        outputs = [ctx.outputs.out],
        mnemonic = "PackageTar",
        use_default_shell_env = True,
    )

# A rule for creating a tar file, see README.md
_real_mini_tar = rule(
    implementation = _mini_tar_impl,
    attrs = {
        "mode": attr.string(default = "0555"),
        "mtime": attr.int(default = -1),
        "out": attr.output(),
        "owner": attr.string(default = "0.0"),
        "package_dir": attr.string(default = "/"),
        "srcs": attr.label_list(allow_files = True),
        "strip_prefix": attr.string(),

        # Implicit dependencies.
        "build_tar": attr.label(
            default = Label("//tools/build_defs/pkg:build_tar"),
            cfg = "exec",
            executable = True,
            allow_files = True,
        ),
    },
)

def mini_tar(name, out = None, **kwargs):
    if not out:
        out = name + ".tar",
    _real_mini_tar(
        name = name,
        out = out,
        **kwargs
    )
