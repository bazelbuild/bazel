# Copyright 2022 The Bazel Authors. All rights reserved.
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
"""A module extension to bring in the remote coverage tools under
@remote_coverage_tools."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# To generate an updated version of CoverageOutputGenerator:
# 1. run tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator/build_and_upload.sh
# 2. Copy the newly uploaded file to https://mirror.bazel.build/bazel_coverage_output_generator/releases.
# 3. Update this file to point to the new release.
def _remote_coverage_tools_extension_impl(ctx):
    http_archive(
        name = "remote_coverage_tools",
        sha256 = "aab349130118497d86bc79e3f735f026d3c36e7d38529063e91da1c29cc2ea47",
        urls = [
            "https://mirror.bazel.build/bazel_coverage_output_generator/releases/coverage_output_generator-v2.9.zip",
        ],
    )
    return ctx.extension_metadata(reproducible = True)

remote_coverage_tools_extension = module_extension(
    implementation = _remote_coverage_tools_extension_impl,
)
