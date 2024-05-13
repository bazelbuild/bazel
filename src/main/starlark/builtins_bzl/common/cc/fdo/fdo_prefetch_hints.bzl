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

"""Starlark implementation of fdo_prefetch_hints rule."""

FdoPrefetchHintsInfo = provider(
    doc = "Contains the profile used for prefetch hints",
    fields = ["artifact"],
)

def _impl(ctx):
    return FdoPrefetchHintsInfo(artifact = ctx.file.profile)

fdo_prefetch_hints = rule(
    implementation = _impl,
    doc = """
<p>Represents an FDO prefetch hints profile that is either in the workspace.
Examples:</p>

<pre><code class="lang-starlark">
fdo_prefetch_hints(
    name = "hints",
    profile = "//path/to/hints:profile.afdo",
)
</code></pre>""",
    attrs = {
        "profile": attr.label(
            allow_single_file = [".afdo"],
            mandatory = True,
            doc = """
Label of the hints profile. The hints file has the .afdo extension
The label can also point to an fdo_absolute_path_profile rule.""",
        ),
    },
    provides = [FdoPrefetchHintsInfo],
)
