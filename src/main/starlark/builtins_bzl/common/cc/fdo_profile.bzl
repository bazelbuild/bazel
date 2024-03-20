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

load(":common/paths.bzl", "paths")

FdoProfileInfo = provider(
    doc = "Contains the profile used for FDO",
    fields = ["artifact", "absolute_path", "proto_profile_artifact", "memprof_artifact"],
)

def _impl(ctx):
    # proto_profile and memprof_profile can only be used if at least one of profile and
    # absolute_path_profile are provided.
    if bool(ctx.file.profile) == bool(ctx.attr.absolute_path_profile):
        fail("exactly one of profile and absolute_path_profile should be specified")
    if ctx.file.proto_profile and not ctx.file.proto_profile.is_source:
        fail("Attribute proto_profile: the target is not an input file")
    if ctx.file.memprof_profile and not ctx.file.memprof_profile.is_source:
        fail("Attribute memprof_profile: the target is not an input file")

    if ctx.attr.profile:
        return FdoProfileInfo(
            artifact = ctx.file.profile,
            proto_profile_artifact = ctx.file.proto_profile,
            memprof_artifact = ctx.file.memprof_profile,
        )
    else:
        if not ctx.fragments.cpp.enable_fdo_profile_absolute_path():
            fail("this attribute cannot be used when --enable_fdo_profile_absolute_path is false")
        absolute_path_profile = ctx.expand_make_variables("absolute_path_profile", ctx.attr.absolute_path_profile, {})
        if not paths.is_absolute(absolute_path_profile):
            fail("Attribute: absolute_path_profile: %s is not an absolute path" % ctx.attr.absolute_path_profile)
        return FdoProfileInfo(
            absolute_path = absolute_path_profile,
            proto_profile_artifact = ctx.file.proto_profile,
            memprof_artifact = ctx.file.memprof_profile,
        )

fdo_profile = rule(
    implementation = _impl,
    doc = """

<p>Represents an FDO profile that is either in the workspace or at a specified absolute path.
Examples:</p>

<pre><code class="lang-starlark">
fdo_profile(
    name = "fdo",
    profile = "//path/to/fdo:profile.zip",
)

fdo_profile(
  name = "fdo_abs",
  absolute_path_profile = "/absolute/path/profile.zip",
)
</code></pre>""",
    attrs = {
        "profile": attr.label(
            allow_single_file = [".profraw", ".profdata", ".zip", ".afdo", ".xfdo"],
            doc = """
Label of the FDO profile or a rule which generates it. The FDO file can have one of the
following extensions: .profraw for unindexed LLVM profile, .profdata for indexed LLVM
profile, .zip that holds an LLVM profraw profile, .afdo for AutoFDO profile, .xfdo for
XBinary profile. The label can also point to an fdo_absolute_path_profile rule.""",
        ),
        "absolute_path_profile": attr.string(
            doc = """
Absolute path to the FDO profile. The FDO file can have one of the following extensions:
.profraw for unindexed LLVM profile, .profdata for indexed LLVM profile, .zip
that holds an LLVM profraw profile, or .afdo for AutoFDO profile.""",
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
    fragments = ["cpp"],
)
