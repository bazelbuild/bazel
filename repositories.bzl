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
    "googleapis",
    "grpc-java",
    "io_bazel_skydoc",
    "platforms",
    "protobuf",
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
        integrity = "sha256-MY0MLtPIdvt+oslSlFzc997PtSZMpRrs4VnmNaxT1UQ=",
        downloaded_file_path = "zulu-linux-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu21.36.17-ca-jdk21.0.4-linux_x64.tar.gz",
    )
    http_file(
        name = "openjdk_linux_aarch64_vanilla",
        integrity = "sha256-2jwtfbM2cLz2ZTJEGut/M9zw0ifI2v5841zuZ/aCnEw=",
        downloaded_file_path = "zulu-linux-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu21.36.17-ca-jdk21.0.4-linux_aarch64.tar.gz",
    )
    http_file(
        name = "openjdk_linux_s390x_vanilla",
        integrity = "sha256-yQDI1k+rHlMnSXT6Skxzalo3VEhaXFb0lHKBSAdzZYo=",
        downloaded_file_path = "adoptopenjdk-s390x-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.4%2B7/OpenJDK21U-jdk_s390x_linux_hotspot_21.0.4_7.tar.gz",
    )
    http_file(
        name = "openjdk_linux_ppc64le_vanilla",
        integrity = "sha256-wgjND7kFYGRKkPkoZn0vU7/kCMlXpeNiBlha2HRCd2E=",
        downloaded_file_path = "adoptopenjdk-ppc64le-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.4%2B7/OpenJDK21U-jdk_ppc64le_linux_hotspot_21.0.4_7.tar.gz",
    )
    http_file(
        name = "openjdk_macos_x86_64_vanilla",
        integrity = "sha256-XOdaaiR8cCm3TEynz29g/SstaM4eiVb7RI0phDFrX+o=",
        downloaded_file_path = "zulu-macos-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu21.36.17-ca-jdk21.0.4-macosx_x64.tar.gz",
    )
    http_file(
        name = "openjdk_macos_aarch64_vanilla",
        integrity = "sha256-vCdQ+BoWbMbpwwroqrpU8lOoyOydjPwEpVX+IHEse/8=",
        downloaded_file_path = "zulu-macos-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu21.36.17-ca-jdk21.0.4-macosx_aarch64.tar.gz",
    )
    http_file(
        name = "openjdk_win_vanilla",
        integrity = "sha256-13Ha0Q0/C0QMNobR89K2izIIAqyXshLYdnGvPy7viEg=",
        downloaded_file_path = "zulu-win-vanilla.zip",
        url = "https://cdn.azul.com/zulu/bin/zulu21.36.17-ca-jdk21.0.4-win_x64.zip",
    )
    http_file(
        name = "openjdk_win_arm64_vanilla",
        integrity = "sha256-n4c+zPAwsdPch57B6w/14Rv3YALcgcXGRMNGK/bFFGs=",
        downloaded_file_path = "zulu-win-arm64.zip",
        url = "https://cdn.azul.com/zulu/bin/zulu21.36.17-ca-jdk21.0.4-win_aarch64.zip",
    )
