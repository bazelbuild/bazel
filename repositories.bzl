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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
load("//src/tools/bzlmod:utils.bzl", "get_canonical_repo_name")

##################################################################################
#
# The list of repositories required while bootstrapping Bazel offline
#
##################################################################################
DIST_ARCHIVE_REPOS = [get_canonical_repo_name(repo) for repo in [
    # keep sorted
    "abseil-cpp",
    "apple_support",
    "bazel_skylib",
    "blake3",
    "c-ares",
    "com_github_grpc_grpc",
    "com_google_protobuf",
    "googleapis",
    "grpc-java",
    "io_bazel_skydoc",
    "platforms",
    "rules_cc",
    "rules_go",
    "rules_graalvm",
    "rules_java",
    "rules_jvm_external",
    "rules_kotlin",
    "rules_license",
    "rules_pkg",
    "rules_proto",
    "rules_python",
    "rules_shell",
    "zlib",
    "zstd-jni",
]] + [(get_canonical_repo_name("com_github_grpc_grpc") + "+grpc_repo_deps_ext+" + suffix) for suffix in [
    # Extra grpc dependencies introduced via its module extension
    "com_envoyproxy_protoc_gen_validate",
    "com_github_cncf_xds",
    "envoy_api",
    "google_cloud_cpp",
    "io_opencensus_cpp",
]] + [
    "bazel_features+",
    "rules_apple+",
    "rules_foreign_cc+",
    "rules_fuzzing+",
    "rules_swift+",
]

##################################################################################
#
# Make sure all URLs below are mirrored to https://mirror.bazel.build
#
##################################################################################

def embedded_jdk_repositories():
    """OpenJDK distributions used to create a version of Bazel bundled with the OpenJDK."""
    http_file(
        name = "openjdk_linux_vanilla",
        integrity = "sha256-rAFnZr8+a29vrXmSXwUQ7rgX6G8DgPvAioHSdU64kJI=",
        downloaded_file_path = "zulu-linux-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu23.30.13-ca-jdk23.0.1-linux_x64.tar.gz",
    )
    http_file(
        name = "openjdk_linux_aarch64_vanilla",
        integrity = "sha256-YsWDS56XxBqVacn8HoiUhPL3bongmOyj5tSkN5RREW8=",
        downloaded_file_path = "zulu-linux-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu23.30.13-ca-jdk23.0.1-linux_aarch64.tar.gz",
    )
    http_file(
        name = "openjdk_linux_s390x_vanilla",
        integrity = "sha256-OUGdcggvrqbSUBIj8cv2qRKLwjAArft7fues/OQiUJw=",
        downloaded_file_path = "adoptopenjdk-s390x-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin23-binaries/releases/download/jdk-23.0.1%2B11/OpenJDK23U-jdk_s390x_linux_hotspot_23.0.1_11.tar.gz",
    )
    http_file(
        name = "openjdk_linux_ppc64le_vanilla",
        integrity = "sha256-GIWrFB/nuO1r63e4FLHByZ/VRxM5m/kX7bakAgVFrd4=",
        downloaded_file_path = "adoptopenjdk-ppc64le-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin23-binaries/releases/download/jdk-23.0.1%2B11/OpenJDK23U-jdk_ppc64le_linux_hotspot_23.0.1_11.tar.gz",
    )
    http_file(
        name = "openjdk_macos_x86_64_vanilla",
        integrity = "sha256-aYyf1SKQGgO5wPb+fBfJg19s+qUVmHvemh6xDwXQtpc=",
        downloaded_file_path = "zulu-macos-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu23.30.13-ca-jdk23.0.1-macosx_x64.tar.gz",
    )
    http_file(
        name = "openjdk_macos_aarch64_vanilla",
        integrity = "sha256-OIXfVgx6ipx3uALCLcGUbNLuEp2d/NdFWKj/npRZ5s8=",
        downloaded_file_path = "zulu-macos-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu23.30.13-ca-jdk23.0.1-macosx_aarch64.tar.gz",
    )
    http_file(
        name = "openjdk_win_vanilla",
        integrity = "sha256-wnqLLUrcL778vxk874CEJJHs1pclQU0w5Z5RR2nzWMw=",
        downloaded_file_path = "zulu-win-vanilla.zip",
        url = "https://cdn.azul.com/zulu/bin/zulu23.30.13-ca-jdk23.0.1-win_x64.zip",
    )

    # Later version of the JDK for Windows ARM64 are not available yet.
    http_file(
        name = "openjdk_win_arm64_vanilla",
        integrity = "sha256-9a1/U5900StiSMD9n0tBZFXc9oA5ALKOjRkFTz3Mbpg=",
        downloaded_file_path = "zulu-win-arm64.zip",
        url = "https://cdn.azul.com/zulu/bin/zulu21.38.21-ca-jdk21.0.5-win_aarch64.zip",
    )
