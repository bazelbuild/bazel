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
)
```
"""

def workspace_and_buildfile(ctx):
  """Utility function for writing WORKSPACE and, if requested, a BUILD file.

  It assumes the paramters name, build_file, and build_file_contents to be
  present in ctx.attr, the latter two possibly with value None.

  Args:
    ctx: The repository context of the repository rule calling this utility
      function.
  """
  if ctx.attr.build_file and ctx.attr.build_file_content:
    ctx.fail("Only one of build_file and build_file_content can be provided.")

  ctx.file("WORKSPACE", "workspace(name = \"{name}\")\n".format(name=ctx.name))

  if ctx.attr.build_file:
    bash_exe = ctx.os.environ["BAZEL_SH"] if "BAZEL_SH" in ctx.os.environ else "bash"
    ctx.execute([bash_exe, "-c", "rm -f BUILD BUILD.bazel"])
    ctx.symlink(ctx.attr.build_file, "BUILD")
  elif ctx.attr.build_file_content:
    bash_exe = ctx.os.environ["BAZEL_SH"] if "BAZEL_SH" in ctx.os.environ else "bash"
    ctx.execute([bash_exe, "-c", "rm -f BUILD.bazel"])
    ctx.file("BUILD", ctx.attr.build_file_content)


