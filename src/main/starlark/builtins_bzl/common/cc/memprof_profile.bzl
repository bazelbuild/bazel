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

"""Starlark implementation of memprof_profile rule."""

load(":common/paths.bzl", "paths")

MemProfProfileInfo = provider(
    doc = "Contains the memprof profile",
    fields = ["artifact", "absolute_path"],
)

def _impl(ctx):
    if bool(ctx.file.profile) == bool(ctx.attr.absolute_path_profile):
        fail("exactly one of profile and absolute_path_profile should be specified")

    if ctx.attr.profile:
        return MemProfProfileInfo(artifact = ctx.file.profile)
    else:
        if not ctx.fragments.cpp.enable_fdo_profile_absolute_path():
            fail("absolute_path_profile cannot be used when --enable_fdo_profile_absolute_path is false")
        if not paths.is_absolute(ctx.attr.absolute_path_profile):
            fail("Attribute: absolute_path_profile: %s is not an absolute path" % ctx.attr.absolute_path_profile)
        return MemProfProfileInfo(absolute_path = ctx.attr.absolute_path_profile)

memprof_profile = rule(
    implementation = _impl,
    doc = """
<p>Represents a MEMPROF profile that is either in the workspace or at a specified
absolute path.
Examples:</p>

<pre><code class="lang-starlark">
memprof_profile(
    name = "memprof",
    profile = "//path/to/memprof:profile.afdo",
)

memprof_profile(
  name = "memprof_abs",
  absolute_path_profile = "/absolute/path/profile.afdo",
)
</code></pre>""",
    attrs = {
        "profile": attr.label(
            allow_single_file = [".profdata", ".zip"],
            doc = """
Label of the MEMPROF profile. The profile is expected to have
either a .profdata extension (for an indexed/symbolized memprof
profile), or a .zip extension for a zipfile containing a memprof.profdata
file.
The label can also point to an fdo_absolute_path_profile rule.""",
        ),
        "absolute_path_profile": attr.string(
            doc = """
Absolute path to the MEMPROF profile. The file may only have a .profdata or
.zip extension (where the zipfile must contain a memprof.profdata file).""",
        ),
    },
    provides = [MemProfProfileInfo],
    fragments = ["cpp"],
)
