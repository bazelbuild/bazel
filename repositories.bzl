# Copyright 2023 The Bazel Authors. All rights reserved.
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
"""Macros for defining dependencies we need to build Bazel.

"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("//src/tools/bzlmod:utils.bzl", "get_canonical_repo_name")

##################################################################################
#
# The list of repositories required while bootstrapping Bazel offline
#
##################################################################################
DIST_ARCHIVE_REPOS = [
    # Bazel module dependencies, keep sorted
    "abseil-cpp+",
    "apple_support+",
    "bazel_features+",
    "bazel_skylib+",
    "blake3+",
    "c-ares+",
    "envoy_api+",
    "googleapis+",
    "googleapis-grpc-java+",
    "googleapis-java+",
    "googleapis-rules-registry+",
    "grpc+",
    "grpc-java+",
    "opencensus-cpp+",
    "platforms",
    "protobuf+",
    "protoc-gen-validate+",
    "re2+",
    "rules_apple+",
    "rules_cc+",
    "rules_fuzzing+",
    "rules_go+",
    "rules_graalvm+",
    "rules_java+",
    "rules_jvm_external+",
    "rules_kotlin+",
    "rules_license+",
    "rules_perl+",
    "rules_pkg+",
    "rules_proto+",
    "rules_python+",
    "rules_shell+",
    "rules_swift+",
    "stardoc+",
    "with_cfg.bzl+",
    "xds+",
    "zlib+",
    "zstd-jni+",
] + [get_canonical_repo_name(repo) for repo in [
    # Module extension repos
    "async_profiler",
    "async_profiler_linux_arm64",
    "async_profiler_linux_x64",
    "async_profiler_macos",
]]

##################################################################################
#
# Make sure all URLs below are mirrored to https://mirror.bazel.build
#
##################################################################################

def embedded_jdk_repositories():
    """OpenJDK distributions used to create a version of Bazel bundled with the OpenJDK."""
    http_file(
        name = "openjdk_linux_vanilla",
        integrity = "sha256-8XUtAFG2yiM2Jd2ywYyRcO2+VcXuZRW/79jqAZfuHCA=",
        downloaded_file_path = "zulu-linux-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu25.32.17-ca-jdk25.0.2-linux_x64.tar.gz",
    )
    http_file(
        name = "openjdk_linux_aarch64_vanilla",
        integrity = "sha256-9ylaux3ssvbg6p92CnCRfy1HNW20Nm1hXHq5sBw8aGY=",
        downloaded_file_path = "zulu-linux-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu25.32.17-ca-jdk25.0.2-linux_aarch64.tar.gz",
    )
    http_file(
        name = "openjdk_linux_ppc64le_vanilla",
        integrity = "sha256-smK3NbIVFzADdm2jZYjV9xfc6toChttBtDn5P7KtpGg=",
        downloaded_file_path = "adoptopenjdk-ppc64le-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.2%2B10/OpenJDK25U-jdk_ppc64le_linux_hotspot_25.0.2_10.tar.gz",
    )
    http_file(
        name = "openjdk_linux_riscv64_vanilla",
        integrity = "sha256-FoEZ5PujUPTms8qSRQorkKhQK4miNaBEFemt+fXTFk4=",
        downloaded_file_path = "adoptopenjdk-riscv64-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.2%2B10/OpenJDK25U-jdk_riscv64_linux_hotspot_25.0.2_10.tar.gz",
    )
    http_file(
        name = "openjdk_linux_s390x_vanilla",
        integrity = "sha256-FeXLytzz1DYjwxuCUGPNwoF7nxuoQLUdxu9w5dM8hOM=",
        downloaded_file_path = "adoptopenjdk-s390x-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.2%2B10/OpenJDK25U-jdk_s390x_linux_hotspot_25.0.2_10.tar.gz",
    )
    http_file(
        name = "openjdk_macos_x86_64_vanilla",
        integrity = "sha256-/iI0bBkpIMW4SyP0qiTj3rsLuU/EKAFvfCssd5DIJ4A=",
        downloaded_file_path = "zulu-macos-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu25.32.17-ca-jdk25.0.2-macosx_x64.tar.gz",
    )
    http_file(
        name = "openjdk_macos_aarch64_vanilla",
        integrity = "sha256-U3rHT6HKLE3Y9gY93t4BOK5KiW8Si/4QtCin3MSqkp8=",
        downloaded_file_path = "zulu-macos-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu25.32.17-ca-jdk25.0.2-macosx_aarch64.tar.gz",
    )
    http_file(
        name = "openjdk_win_vanilla",
        integrity = "sha256-kLy7y+L7euxDzo/S76Ak/MSPALvc16drWPkS1rl+bVY=",
        downloaded_file_path = "zulu-win-vanilla.zip",
        url = "https://cdn.azul.com/zulu/bin/zulu25.32.17-ca-jdk25.0.2-win_x64.zip",
    )
    http_file(
        name = "openjdk_win_arm64_vanilla",
        integrity = "sha256-X0R5KhKvIQC025tRIMJ8Qq+AgK1jvJ6q3JyiXlx91ZA=",
        downloaded_file_path = "zulu-win-arm64.zip",
        url = "https://cdn.azul.com/zulu/bin/zulu25.32.17-ca-jdk25.0.2-win_aarch64.zip",
    )

def _async_profiler_repos(ctx):
    http_file(
        name = "async_profiler",
        downloaded_file_path = "async-profiler.jar",
        integrity = "sha256-d2VI3O7jJpa197ArxjzAixnb/nTciR6X/j4p4H+qeMw=",
        urls = ["https://github.com/async-profiler/async-profiler/releases/download/v4.1/async-profiler.jar"],
    )

    _ASYNC_PROFILER_BUILD_TEMPLATE = """
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")

copy_file(
    name = "libasyncProfiler",
    src = "libasyncProfiler.{ext}",
    out = "{tag}/libasyncProfiler.so",
    visibility = ["//visibility:public"],
)
"""

    http_archive(
        name = "async_profiler_linux_arm64",
        build_file_content = _ASYNC_PROFILER_BUILD_TEMPLATE.format(
            ext = "so",
            tag = "linux-arm64",
        ),
        integrity = "sha256-0Mucl8OAZytiXAblo+1XjpkPRnTGqui1JJ9YTEyaxQ4=",
        strip_prefix = "async-profiler-4.1-linux-arm64/lib",
        urls = ["https://github.com/async-profiler/async-profiler/releases/download/v4.1/async-profiler-4.1-linux-arm64.tar.gz"],
    )

    http_archive(
        name = "async_profiler_linux_x64",
        build_file_content = _ASYNC_PROFILER_BUILD_TEMPLATE.format(
            ext = "so",
            tag = "linux-x64",
        ),
        integrity = "sha256-OxOjigBj9pcNmFo3ndrtkbzzfiOaHqRh0J6s9inz3eE=",
        strip_prefix = "async-profiler-4.1-linux-x64/lib",
        urls = ["https://github.com/async-profiler/async-profiler/releases/download/v4.1/async-profiler-4.1-linux-x64.tar.gz"],
    )

    http_archive(
        name = "async_profiler_macos",
        build_file_content = _ASYNC_PROFILER_BUILD_TEMPLATE.format(
            ext = "dylib",
            tag = "macos",
        ),
        integrity = "sha256-xfsFjiEiguk4SiYDGgURn183UMdVsrX7bQimtngD6tA=",
        strip_prefix = "async-profiler-4.1-macos/lib",
        urls = ["https://github.com/async-profiler/async-profiler/releases/download/v4.1/async-profiler-4.1-macos.zip"],
    )

# This is an extension (instead of use_repo_rule usages) only to create a
# lockfile entry for the distribution repo module extension.
async_profiler_repos = module_extension(_async_profiler_repos)
