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

PropellerOptimizeInfo = provider(
    doc = "Contains the profile used for propeller",
    fields = ["cc_profile", "ld_profile"],
)

def _impl(ctx):
    return PropellerOptimizeInfo(
        cc_profile = ctx.file.cc_profile,
        ld_profile = ctx.file.ld_profile,
    )

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
</code></pre>""",
    attrs = {
        "cc_profile": attr.label(
            allow_single_file = [".txt"],
            mandatory = True,
            doc = """
Label of the profile passed to the various compile actions.  This file has
the .txt extension.""",
        ),
        "ld_profile": attr.label(
            allow_single_file = [".txt"],
            mandatory = True,
            doc = """
Label of the profile passed to the link action.  This file has
the .txt extension.""",
        ),
    },
    provides = [PropellerOptimizeInfo],
)
