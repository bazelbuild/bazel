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
"""Module extensions for tools that are downloaded rather than embedded into
@bazel_tools."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//tools:build_defs.bzl", "PREBUILT_TOOLS_PLATFORMS")

# To generate an updated version of CoverageOutputGenerator:
# 1. run tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator/build_and_upload.sh
# 2. Copy the newly uploaded file to https://mirror.bazel.build/bazel_coverage_output_generator/releases.
# 3. Update this file to point to the new release.
def _remote_coverage_tools_extension_impl(ctx):
    http_archive(
        name = "remote_coverage_tools",
        sha256 = "089dd102b8c4ac656932080a9b531c28ed5f1b88db0e2524c736f0f0ee0d130b",
        urls = [
            "https://mirror.bazel.build/bazel_coverage_output_generator/releases/coverage_output_generator-v3.0.zip",
        ],
    )
    return ctx.extension_metadata(reproducible = True)

remote_coverage_tools_extension = module_extension(
    implementation = _remote_coverage_tools_extension_impl,
)

# TODO: Fill in after the first release. To release a new version:
# 1. Run the prebuilt_tools Buildkite pipeline, which runs
#    src/prebuilt_tools/upload.sh for every platform in PREBUILT_TOOLS_PLATFORMS.
# 2. Run src/prebuilt_tools/release.sh <commit hash> <version> and replace the
#    version and hashes below with its output.
_PREBUILT_TOOLS_VERSION = "TODO"
_PREBUILT_TOOLS_SHA256 = {platform: "TODO" for platform in PREBUILT_TOOLS_PLATFORMS}

def _prebuilt_tools_extension_impl(ctx):
    for platform in PREBUILT_TOOLS_PLATFORMS:
        http_archive(
            name = "prebuilt_tools_" + platform,
            sha256 = _PREBUILT_TOOLS_SHA256[platform],
            urls = [
                "https://mirror.bazel.build/bazel_prebuilt_tools/releases/v{version}/prebuilt_tools_{platform}-v{version}.zip".format(
                    platform = platform,
                    version = _PREBUILT_TOOLS_VERSION,
                ),
            ],
        )
    return ctx.extension_metadata(reproducible = True)

prebuilt_tools_extension = module_extension(
    implementation = _prebuilt_tools_extension_impl,
)
