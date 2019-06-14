# Copyright 2019 The Bazel Authors. All rights reserved.
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

"""write_file() rule from bazel-skylib 0.8.0.

This file is a copy of rules/private/write_file_private.bzl [1] with some edits:
  - this DocString is different
  - the rule's 'out' attribute does not create a label, so it is select()-able

IMPORTANT: please do not use this rule outside of this package.
Related discussion here [2].


[1] https://github.com/bazelbuild/bazel-skylib/blob/3721d32c14d3639ff94320c780a60a6e658fb033/rules/private/write_file_private.bzl

[2] https://groups.google.com/d/msg/bazel-dev/I8IvJyoyo-s/AttqDcnOCgAJ
"""

def _common_impl(ctx, is_executable):
    # ctx.actions.write creates a FileWriteAction which uses UTF-8 encoding.
    out = ctx.actions.declare_file(ctx.attr.out)
    ctx.actions.write(
        output = out,
        content = "\n".join(ctx.attr.content) if ctx.attr.content else "",
        is_executable = is_executable,
    )
    files = depset(direct = [out])
    runfiles = ctx.runfiles(files = [out])
    if is_executable:
        return [DefaultInfo(files = files, runfiles = runfiles, executable = out)]
    else:
        return [DefaultInfo(files = files, runfiles = runfiles)]

def _impl(ctx):
    return _common_impl(ctx, False)

def _ximpl(ctx):
    return _common_impl(ctx, True)

_ATTRS = {
    "content": attr.string_list(mandatory = False, allow_empty = True),
    "out": attr.string(mandatory = True),
}

_write_file = rule(
    implementation = _impl,
    provides = [DefaultInfo],
    attrs = _ATTRS,
)

_write_xfile = rule(
    implementation = _ximpl,
    executable = True,
    provides = [DefaultInfo],
    attrs = _ATTRS,
)

def write_file(name, out, content = [], is_executable = False, **kwargs):
    """Creates a UTF-8 encoded text file.

    Args:
      name: Name of the rule.
      out: Path of the output file, relative to this package.
      content: A list of strings. Lines of text, the contents of the file.
          Newlines are added automatically after every line except the last one.
      is_executable: A boolean. Whether to make the output file executable. When
          True, the rule's output can be executed using `bazel run` and can be
          in the srcs of binary and test rules that require executable sources.
      **kwargs: further keyword arguments, e.g. `visibility`
    """
    if is_executable:
        _write_xfile(
            name = name,
            content = content,
            out = out,
            **kwargs
        )
    else:
        _write_file(
            name = name,
            content = content,
            out = out,
            **kwargs
        )
