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

MemProfProfileInfo = provider(
    doc = "Contains the memprof profile",
    fields = ["artifact"],
)

def _impl(ctx):
    return MemProfProfileInfo(artifact = ctx.file.profile)

memprof_profile = rule(
    implementation = _impl,
    doc = """
<p>Represents a MEMPROF profile that is in the workspace.
Example:</p>

<pre><code class="lang-starlark">
memprof_profile(
    name = "memprof",
    profile = "//path/to/memprof:profile.afdo",
)

</code></pre>""",
    attrs = {
        "profile": attr.label(
            allow_single_file = [".profdata", ".zip"],
            mandatory = True,
            doc = """
Label of the MEMPROF profile. The profile is expected to have
either a .profdata extension (for an indexed/symbolized memprof
profile), or a .zip extension for a zipfile containing a memprof.profdata
file.
The label can also point to an fdo_absolute_path_profile rule.""",
        ),
    },
    provides = [MemProfProfileInfo],
)
