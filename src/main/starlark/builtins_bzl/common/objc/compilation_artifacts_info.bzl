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

"""Artifacts related to compilation."""

# File types that should actually be compiled, or that are already compiled.
# Important: this tuple must not contain any header file extensions. See comment where it's used.
COMPILABLE_OR_PRECOMPILED_SRC_EXTENSIONS = (
    "cc",  # CPP extensions
    "cpp",
    "mm",
    "cxx",
    "C",
    "m",  # Non-CPP extensions
    "c",
    "s",  # Assembly extensions
    "S",
    "asm",
    "o",  # Object file extensions
)

def _compilation_artifacts_init(
        ctx = None,
        srcs = None,
        non_arc_srcs = None,
        hdrs = None,
        intermediate_artifacts = None):
    if ctx == None and srcs == None and non_arc_srcs == None and hdrs == None and intermediate_artifacts == None:
        return {
            "srcs": [],
            "non_arc_srcs": [],
            "additional_hdrs": [],
            "archive": None,
        }
    if ctx != None and (srcs != None or non_arc_srcs != None or hdrs != None):
        fail("CompilationArtifactsInfo() params ctx and (srcs, non_arc_srcs, hdrs) are mutually exclusive")
    if ctx != None:
        srcs = ctx.files.srcs if hasattr(ctx.files, "srcs") else []
        non_arc_srcs = ctx.files.non_arc_srcs if hasattr(ctx.files, "non_arc_srcs") else []
        hdrs = []

    # Note: the condition under which we set an archive artifact needs to match the condition for
    # which we create the archive in compilation_support.bzl.  In particular, if srcs are all
    # headers, we don't generate an archive.
    if (
        non_arc_srcs or
        [s for s in srcs if s.is_directory or
                            s.extension in COMPILABLE_OR_PRECOMPILED_SRC_EXTENSIONS]
    ):
        archive = intermediate_artifacts.archive()
    else:
        archive = None
    return {
        "srcs": srcs,
        "non_arc_srcs": non_arc_srcs,
        "additional_hdrs": hdrs,
        "archive": archive,
    }

CompilationArtifactsInfo, _new_compilationartifactsinfo = provider(
    "Any rule containing compilable sources will create an instance of this provider.",
    fields = ["srcs", "non_arc_srcs", "additional_hdrs", "archive"],
    init = _compilation_artifacts_init,
)
