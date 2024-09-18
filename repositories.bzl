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
DIST_ARCHIVE_REPOS = [get_canonical_repo_name(repo) for repo in [
    # keep sorted
    "abseil-cpp",
    "apple_support",
    "bazel_skylib",
    "blake3",
    "c-ares",
    "com_github_grpc_grpc",
    "com_google_protobuf",
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
    "sqlite3",
    "upb",
    "zlib",
    "zstd-jni",
]] + [(get_canonical_repo_name("com_github_grpc_grpc") + suffix) for suffix in [
    # Extra grpc dependencies introduced via its module extension
    "+grpc_repo_deps_ext+bazel_gazelle",  # TODO: Should be a bazel_dep
    "+grpc_repo_deps_ext+bazel_skylib",  # TODO: Should be removed
    "+grpc_repo_deps_ext+com_envoyproxy_protoc_gen_validate",
    "+grpc_repo_deps_ext+com_github_cncf_udpa",
    "+grpc_repo_deps_ext+com_google_googleapis",
    "+grpc_repo_deps_ext+envoy_api",
    "+grpc_repo_deps_ext+rules_cc",  # TODO: Should be removed
]] + ["bazel_features+"]

##################################################################################
#
# Make sure all URLs below are mirrored to https://mirror.bazel.build
#
##################################################################################

def embedded_jdk_repositories():
    """OpenJDK distributions used to create a version of Bazel bundled with the OpenJDK."""
    http_file(
        name = "openjdk_linux_vanilla",
        integrity = "sha256-a1YPqBMaWkoruNFoSckLyx00LCOZNsowlSn2L3XCDJA=",
        downloaded_file_path = "zulu-linux-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu23.28.85-ca-jdk23.0.0-linux_x64.tar.gz",
    )
    http_file(
        name = "openjdk_linux_aarch64_vanilla",
        integrity = "sha256-/i+ch7BMAwMQ1C4e3shp9BHuQ67vVXfmIK1YKs7L24M=",
        downloaded_file_path = "zulu-linux-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu23.28.85-ca-jdk23.0.0-linux_aarch64.tar.gz",
    )

    # Later versions of the JDK are not available for s390x and ppc64le from any vendor at this point.
    http_file(
        name = "openjdk_linux_s390x_vanilla",
        integrity = "sha256-RlJ8/FYFUvBcBGJSDWnUOPFEo9yCBmh5Ujh8kQzdTEA=",
        downloaded_file_path = "adoptopenjdk-s390x-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin22-binaries/releases/download/jdk-22.0.2%2B9/OpenJDK22U-jdk_s390x_linux_hotspot_22.0.2_9.tar.gz",
    )
    http_file(
        name = "openjdk_linux_ppc64le_vanilla",
        integrity = "sha256-L1wZBZV6J4Iyu9w5vUCOe9FyyJRFSCSZYg0v6VEL1Po=",
        downloaded_file_path = "adoptopenjdk-ppc64le-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin22-binaries/releases/download/jdk-22.0.2%2B9/OpenJDK22U-jdk_ppc64_aix_hotspot_22.0.2_9.tar.gz",
    )
    http_file(
        name = "openjdk_macos_x86_64_vanilla",
        integrity = "sha256-rEr8M3KF9Z95gV8sHqi5lQD2RJjtssZx8Q8goy6danw=",
        downloaded_file_path = "zulu-macos-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu23.28.85-ca-jdk23.0.0-macosx_x64.tar.gz",
    )
    http_file(
        name = "openjdk_macos_aarch64_vanilla",
        integrity = "sha256-gFvfJL0RQgIOATLTMdfa+fStUCrdHYC3rxy0j5eNVDc=",
        downloaded_file_path = "zulu-macos-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu23.28.85-ca-jdk23.0.0-macosx_aarch64.tar.gz",
    )
    http_file(
        name = "openjdk_win_vanilla",
        integrity = "sha256-V3ih8NqHAYijOFw+PKE86qnBEy+3ulGJhwjISgu4CJE=",
        downloaded_file_path = "zulu-win-vanilla.zip",
        url = "https://cdn.azul.com/zulu/bin/zulu23.28.85-ca-jdk23.0.0-win_x64.zip",
    )

    # Later versions of the JDK are not available for Windows ARM64 from any vendor at this point.
    http_file(
        name = "openjdk_win_arm64_vanilla",
        integrity = "sha256-n4c+zPAwsdPch57B6w/14Rv3YALcgcXGRMNGK/bFFGs=",
        downloaded_file_path = "zulu-win-arm64.zip",
        url = "https://cdn.azul.com/zulu/bin/zulu21.36.17-ca-jdk21.0.4-win_aarch64.zip",
    )

def android_deps_repos():
    """Required by building the android tools."""
    http_archive(
        name = "desugar_jdk_libs",
        sha256 = "ef71be474fbb3b3b7bd70cda139f01232c63b9e1bbd08c058b00a8d538d4db17",
        strip_prefix = "desugar_jdk_libs-24dcd1dead0b64aae3d7c89ca9646b5dc4068009",
        url = "https://github.com/google/desugar_jdk_libs/archive/24dcd1dead0b64aae3d7c89ca9646b5dc4068009.zip",
    )
