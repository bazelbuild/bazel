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
"""Utils for manipulating external repositories, once fetched.

### Setup

These utility are intended to be used by other repository rules. They
can be loaded as follows.

```python
load(
    "@bazel_tools//tools/build_defs/repo:utils.bzl",
    "workspace_and_buildfile",
    "patch",
    "update_attrs",
)
```
"""

def workspace_and_buildfile(ctx):
    """Utility function for writing WORKSPACE and, if requested, a BUILD file.

    It assumes the parameters name, build_file, build_file_contents,
    workspace_file, and workspace_file_content to be
    present in ctx.attr, the latter four possibly with value None.

    Args:
      ctx: The repository context of the repository rule calling this utility
        function.
    """
    if ctx.attr.build_file and ctx.attr.build_file_content:
        ctx.fail("Only one of build_file and build_file_content can be provided.")

    if ctx.attr.workspace_file and ctx.attr.workspace_file_content:
        ctx.fail("Only one of workspace_file and workspace_file_content can be provided.")

    if ctx.attr.workspace_file:
        bash_exe = ctx.os.environ["BAZEL_SH"] if "BAZEL_SH" in ctx.os.environ else "bash"
        ctx.execute([bash_exe, "-c", "rm -f WORKSPACE"])
        ctx.symlink(ctx.attr.workspace_file, "WORKSPACE")
    elif ctx.attr.workspace_file_content:
        bash_exe = ctx.os.environ["BAZEL_SH"] if "BAZEL_SH" in ctx.os.environ else "bash"
        ctx.execute([bash_exe, "-c", "rm -f WORKSPACE"])
        ctx.file("WORKSPACE", ctx.attr.workspace_file_content)
    else:
        ctx.file("WORKSPACE", "workspace(name = \"{name}\")\n".format(name = ctx.name))

    if ctx.attr.build_file:
        bash_exe = ctx.os.environ["BAZEL_SH"] if "BAZEL_SH" in ctx.os.environ else "bash"
        ctx.execute([bash_exe, "-c", "rm -f BUILD BUILD.bazel"])
        ctx.symlink(ctx.attr.build_file, "BUILD.bazel")
    elif ctx.attr.build_file_content:
        bash_exe = ctx.os.environ["BAZEL_SH"] if "BAZEL_SH" in ctx.os.environ else "bash"
        ctx.execute([bash_exe, "-c", "rm -f BUILD.bazel"])
        ctx.file("BUILD.bazel", ctx.attr.build_file_content)

def patch(ctx):
    """Implementation of patching an already extracted repository"""
    bash_exe = ctx.os.environ["BAZEL_SH"] if "BAZEL_SH" in ctx.os.environ else "bash"
    if len(ctx.attr.patches) > 0 or len(ctx.attr.patch_cmds) > 0:
        ctx.report_progress("Patching repository")
    for patchfile in ctx.attr.patches:
        command = "{patchtool} {patch_args} < {patchfile}".format(
            patchtool = ctx.attr.patch_tool,
            patchfile = ctx.path(patchfile),
            patch_args = " ".join([
                "'%s'" % arg
                for arg in ctx.attr.patch_args
            ]),
        )
        st = ctx.execute([bash_exe, "-c", command])
        if st.return_code:
            fail("Error applying patch %s:\n%s%s" %
                 (str(patchfile), st.stderr, st.stdout))
    for cmd in ctx.attr.patch_cmds:
        st = ctx.execute([bash_exe, "-c", cmd])
        if st.return_code:
            fail("Error applying patch command %s:\n%s%s" %
                 (cmd, st.stdout, st.stderr))

def update_attrs(orig, keys, override):
    """Utility function for altering and adding the specified attributes to a particular repository rule invocation.

     This is used to make a rule reproducible.

    Args:
        orig: dict of actually set attributes (either explicitly or implicitly)
            by a particular rule invocation
        keys: complete set of attributes defined on this rule
        override: dict of attributes to override or add to orig

    Returns:
        dict of attributes with the keys from override inserted/updated
    """
    result = {}
    for key in keys:
        if getattr(orig, key) != None:
            result[key] = getattr(orig, key)
    result["name"] = orig.name
    result.update(override)
    return result

def is_windows(ctx):
    return ctx.os.name.lower().startswith("win")

def remove_dir(ctx, dir_):
    if not dir_.exists:
        return
    remove_cmd = ["rd", "/s", "/q", dir_] if is_windows(ctx) else ["rm", "-rf", dir_]

    # Specify working directory as the parent of the directory that we are removing;
    # otherwise it is going to fail on Windows.
    st = ctx.execute(
        remove_cmd,
        environment = ctx.os.environ,
        working_directory = str(dir_.dirname),
    )
    if st.return_code != 0:
        fail("error removing directory '%s': %s" % (dir_, st.stderr))
