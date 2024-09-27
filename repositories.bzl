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
    "upb",
    "zlib",
    "zstd-jni",
]] + [(get_canonical_repo_name("com_github_grpc_grpc") + suffix) for suffix in [
    # Extra grpc dependencies introduced via its module extension
    "~grpc_repo_deps_ext~bazel_gazelle",  # TODO: Should be a bazel_dep
    "~grpc_repo_deps_ext~bazel_skylib",  # TODO: Should be removed
    "~grpc_repo_deps_ext~com_envoyproxy_protoc_gen_validate",
    "~grpc_repo_deps_ext~com_github_cncf_udpa",
    "~grpc_repo_deps_ext~com_google_googleapis",
    "~grpc_repo_deps_ext~envoy_api",
    "~grpc_repo_deps_ext~rules_cc",  # TODO: Should be removed
]] + [
    # TODO(pcloudy): Remove after https://github.com/bazelbuild/rules_kotlin/issues/1106 is fixed
    get_canonical_repo_name("rules_kotlin") + "~rules_kotlin_extensions~com_github_jetbrains_kotlin",
] + ["bazel_features~"]

##################################################################################
#
# Make sure all URLs below are mirrored to https://mirror.bazel.build
#
##################################################################################

def embedded_jdk_repositories():
    """OpenJDK distributions used to create a version of Bazel bundled with the OpenJDK."""
    http_file(
        name = "openjdk_linux_vanilla",
        sha256 = "0c0eadfbdc47a7ca64aeab51b9c061f71b6e4d25d2d87674512e9b6387e9e3a6",
        downloaded_file_path = "zulu-linux-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu21.28.85-ca-jdk21.0.0-linux_x64.tar.gz",
    )
    http_file(
        name = "openjdk_linux_aarch64_vanilla",
        sha256 = "1fb64b8036c5d463d8ab59af06bf5b6b006811e6012e3b0eb6bccf57f1c55835",
        downloaded_file_path = "zulu-linux-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu21.28.85-ca-jdk21.0.0-linux_aarch64.tar.gz",
    )

    # JDK21 unavailable so use JDK19 instead for linux s390x.
    http_file(
        name = "openjdk_linux_s390x_vanilla",
        sha256 = "f2512f9a8e9847dd5d3557c39b485a8e7a1ef37b601dcbcb748d22e49f44815c",
        downloaded_file_path = "adoptopenjdk-s390x-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin19-binaries/releases/download/jdk-19.0.2%2B7/OpenJDK19U-jdk_s390x_linux_hotspot_19.0.2_7.tar.gz",
    )

    # JDK21 unavailable so use JDK19 instead for linux ppc64le.
    http_file(
        name = "openjdk_linux_ppc64le_vanilla",
        sha256 = "45dde71faf8cbb78fab3c976894259655c8d3de827347f23e0ebe5710921dded",
        downloaded_file_path = "adoptopenjdk-ppc64le-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin20-binaries/releases/download/jdk-20%2B36/OpenJDK20U-jdk_ppc64le_linux_hotspot_20_36.tar.gz",
    )
    http_file(
        name = "openjdk_macos_x86_64_vanilla",
        sha256 = "9639b87db586d0c89f7a9892ae47f421e442c64b97baebdff31788fbe23265bd",
        downloaded_file_path = "zulu-macos-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu21.28.85-ca-jdk21.0.0-macosx_x64.tar.gz",
    )
    http_file(
        name = "openjdk_macos_aarch64_vanilla",
        sha256 = "2a7a99a3ea263dbd8d32a67d1e6e363ba8b25c645c826f5e167a02bbafaff1fa",
        downloaded_file_path = "zulu-macos-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu21.28.85-ca-jdk21.0.0-macosx_aarch64.tar.gz",
    )
    http_file(
        name = "openjdk_win_vanilla",
        sha256 = "e9959d500a0d9a7694ac243baf657761479da132f0f94720cbffd092150bd802",
        downloaded_file_path = "zulu-win-vanilla.zip",
        url = "https://cdn.azul.com/zulu/bin/zulu21.28.85-ca-jdk21.0.0-win_x64.zip",
    )

    # JDK21 unavailable from zulu, we'll use Microsoft's OpenJDK build instead.
    http_file(
        name = "openjdk_win_arm64_vanilla",
        sha256 = "975603e684f2ec5a525b3b5336d6aa0b09b5b7d2d0d9e271bd6a9892ad550181",
        downloaded_file_path = "zulu-win-arm64.zip",
        url = "https://aka.ms/download-jdk/microsoft-jdk-21.0.0-windows-aarch64.zip",
    )

def bazelci_rules_repo():
    """Required by the Bazel CI jobs."""
    http_archive(
        name = "bazelci_rules",
        sha256 = "eca21884e6f66a88c358e580fd67a6b148d30ab57b1680f62a96c00f9bc6a07e",
        strip_prefix = "bazelci_rules-1.0.0",
        url = "https://github.com/bazelbuild/continuous-integration/releases/download/rules-1.0.0/bazelci_rules-1.0.0.tar.gz",
    )

def android_deps_repos():
    """Required by building the android tools."""
    http_archive(
        name = "desugar_jdk_libs",
        sha256 = "ef71be474fbb3b3b7bd70cda139f01232c63b9e1bbd08c058b00a8d538d4db17",
        strip_prefix = "desugar_jdk_libs-24dcd1dead0b64aae3d7c89ca9646b5dc4068009",
        url = "https://github.com/google/desugar_jdk_libs/archive/24dcd1dead0b64aae3d7c89ca9646b5dc4068009.zip",
    )
