# Copyright 2024 The Bazel Authors. All rights reserved.
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

"""Starlark implementation of propeller_optimize rule."""

load(":common/paths.bzl", "paths")

PropellerOptimizeInfo = provider(
    doc = "Contains the profile used for propeller",
    fields = ["cc_profile", "ld_profile"],
)

def _impl(ctx):
    if ctx.file.cc_profile and ctx.attr.absolute_cc_profile:
        fail("Attribute cc_profile: Both relative and absolute profiles are provided.")
    if ctx.file.ld_profile and ctx.attr.absolute_ld_profile:
        fail("Attribute ld_profile: Both relative and absolute profiles are provided.")

    cc_profile = None
    if ctx.file.cc_profile:
        cc_profile = ctx.file.cc_profile
    if ctx.attr.absolute_cc_profile:
        if not ctx.fragments.cpp.enable_fdo_profile_absolute_path():
            fail("absolute paths cannot be used when --enable_fdo_profile_absolute_path is false")
        if not paths.is_absolute(ctx.attr.absolute_cc_profile):
            fail("Attribute: cc_profile: %s is not an absolute path" % ctx.attr.absolute_cc_profile)
        cc_profile = ctx.actions.declare_symlink(ctx.label.name + "/" + paths.basename(ctx.attr.absolute_cc_profile))
        ctx.actions.symlink(
            target_path = ctx.attr.absolute_cc_profile,
            output = cc_profile,
            progress_message = "Symlinking LLVM Propeller Profile " + ctx.attr.absolute_cc_profile,
        )

    ld_profile = None
    if ctx.file.ld_profile:
        ld_profile = ctx.file.ld_profile
    if ctx.attr.absolute_ld_profile:
        if not ctx.fragments.cpp.enable_fdo_profile_absolute_path():
            fail("absolute paths cannot be used when --enable_fdo_profile_absolute_path is false")
        if not paths.is_absolute(ctx.attr.absolute_ld_profile):
            fail("Attribute: ld_profile: %s is not an absolute path" % ctx.attr.absolute_ld_profile)
        ld_profile = ctx.actions.declare_symlink(ctx.label.name + "/" + paths.basename(ctx.attr.absolute_ld_profile))
        ctx.actions.symlink(
            target_path = ctx.attr.absolute_ld_profile,
            output = ld_profile,
            progress_message = "Symlinking LLVM Propeller Profile " + ctx.attr.absolute_ld_profile,
        )

    return PropellerOptimizeInfo(cc_profile = cc_profile, ld_profile = ld_profile)

propeller_optimize = rule(
    implementation = _impl,
    doc = """
<p>Represents a Propeller optimization profile in the workspace.
Example:</p>

<pre><code class="lang-starlark">
propeller_optimize(
    name = "layout",
    cc_profile = "//path:cc_profile.txt",
    ld_profile = "//path:ld_profile.txt"
)

propeller_optimize(
    name = "layout_absolute",
    absolute_cc_profile = "/absolute/cc_profile.txt",
    absolute_ld_profile = "/absolute/ld_profile.txt"
)
</code></pre>""",
    attrs = {
        "cc_profile": attr.label(
            allow_single_file = [".txt"],
            doc = """
Label of the profile passed to the various compile actions.  This file has
the .txt extension.""",
        ),
        "ld_profile": attr.label(
            allow_single_file = [".txt"],
            doc = """
Label of the profile passed to the link action.  This file has
the .txt extension.""",
        ),
        "absolute_cc_profile": attr.string(
            doc = """
Label of the absolute profile passed to the various compile actions.  This file has
the .txt extension.""",
        ),
        "absolute_ld_profile": attr.string(
            doc = """
Label of the absolute profile passed to the various link actions.  This file has
the .txt extension.""",
        ),
    },
    provides = [PropellerOptimizeInfo],
    fragments = ["cpp"],
)
