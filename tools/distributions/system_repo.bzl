# Copyright 2020 The Bazel Authors. All rights reserved.
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
"""Repository rule for providing system libraries for Bazel build."""

def _system_repo_impl(ctx):
    symlinks = ctx.attr.symlinks
    for link in symlinks:
        target = symlinks[link]
        ctx.symlink(target, link)

    ctx.file("WORKSPACE", "workspace(name = \"{name}\")\n".format(name = ctx.name))
    ctx.file("BUILD.bazel", ctx.read(ctx.attr.build_file))

system_repo = repository_rule(
    implementation = _system_repo_impl,
    attrs = {
        "symlinks": attr.string_dict(
            doc = """
                Symlinks to create for this system repo. The key is the link path under this repo,
                the value should be an absolute target path on the system that we want to link.
            """,
        ),
        "build_file": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "The file to use as the BUILD file for this repository.",
        ),
    },
    doc = "A repository rule for providing system libraries for Bazel build",
)
