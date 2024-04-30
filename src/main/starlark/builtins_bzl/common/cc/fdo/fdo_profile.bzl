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

"""Starlark implementation of fdo_profile rule."""

FdoProfileInfo = provider(
    doc = "Contains the profile used for FDO",
    fields = ["artifact", "absolute_path", "proto_profile_artifact", "memprof_artifact"],
)

def _impl(ctx):
    return FdoProfileInfo(
        artifact = ctx.file.profile,
        proto_profile_artifact = ctx.file.proto_profile,
        memprof_artifact = ctx.file.memprof_profile,
    )

fdo_profile = rule(
    implementation = _impl,
    doc = """

<p>Represents an FDO profile that is in the workspace.
Example:</p>

<pre><code class="lang-starlark">
fdo_profile(
    name = "fdo",
    profile = "//path/to/fdo:profile.zip",
)
</code></pre>""",
    attrs = {
        "profile": attr.label(
            allow_single_file = [".profraw", ".profdata", ".zip", ".afdo", ".xfdo"],
            mandatory = True,
            doc = """
Label of the FDO profile or a rule which generates it. The FDO file can have one of the
following extensions: .profraw for unindexed LLVM profile, .profdata for indexed LLVM
profile, .zip that holds an LLVM profraw profile, .afdo for AutoFDO profile, .xfdo for
XBinary profile. The label can also point to an fdo_absolute_path_profile rule.""",
        ),
        "proto_profile": attr.label(allow_single_file = True, doc = """
Label of the protobuf profile."""),
        "memprof_profile": attr.label(allow_single_file = [".profdata", ".zip"], doc = """
Label of the MemProf profile. The profile is expected to have
either a .profdata extension (for an indexed/symbolized memprof
profile), or a .zip extension for a zipfile containing a memprof.profdata
file."""),
    },
    provides = [FdoProfileInfo],
)
