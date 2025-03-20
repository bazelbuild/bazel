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

# Filetype to restrict inputs
tar_filetype = [".tar", ".tar.gz", ".tgz", ".tar.bz2"]

def _quote(filename, protect = "="):
    """Quote the filename, by escaping = by \\= and \\ by \\\\"""
    return filename.replace("\\", "\\\\").replace(protect, "\\" + protect)

def _mini_tar_impl(ctx):
    """Implementation of the mini_tar rule."""

    to_strip = ctx.label.package + "/"

    def dest_path(file):
        # print('FILE', file.path, file.short_path)
        ret = file.short_path
        if ret.startswith(to_strip):
            ret = ret[len(to_strip):]
        return ret

    # Start building the arguments.
    args = ctx.actions.args()
    args.add("--output", ctx.outputs.out.path)
    args.add("--mode", ctx.attr.mode)
    args.add("--owner", ctx.attr.owner)
    if ctx.attr.package_dir:
        args.add("--directory", ctx.attr.package_dir)
    if ctx.attr.mtime != -1:  # Note: Must match default in rule def.
        args.append("--mtime=%d" % ctx.attr.mtime)

    file_inputs = ctx.files.srcs[:]
    for f in file_inputs:
        args.add("--file=%s=%s" % (_quote(f.path), dest_path(f)))
    args.set_param_file_format("flag_per_line")
    args.use_param_file("@%s", use_always = False)
    ctx.actions.run(
        inputs = file_inputs,
        executable = ctx.executable._mini_tar,
        arguments = [args],
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
        "package_dir": attr.string(),
        "srcs": attr.label_list(allow_files = True),

        # Implicit dependencies.
        "_mini_tar": attr.label(
            default = Label("//tools/mini_tar:mini_tar"),
            cfg = "exec",
            executable = True,
            allow_files = True,
        ),
    },
)

def mini_tar(name, out = None, **kwargs):
    if not out:
        out = name + ".tar"
    _real_mini_tar(
        name = name,
        out = out,
        **kwargs
    )
