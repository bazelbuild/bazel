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
    "async_profiler",
    "async_profiler_linux_arm64",
    "async_profiler_linux_x64",
    "async_profiler_macos",
    "bazel_skylib",
    "blake3",
    "c-ares",
    "chicory",
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
        integrity = "sha256-Fk2QHlokC4wYUW9atVvBH8lomrboKQRa6oRnNW3Ns0A=",
        downloaded_file_path = "zulu-linux-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu25.28.85-ca-jdk25.0.0-linux_x64.tar.gz",
    )
    http_file(
        name = "openjdk_linux_aarch64_vanilla",
        integrity = "sha256-tg651UyXukFZVHg0qYzF0BYoHdKz5g50dcukkRMkvLQ=",
        downloaded_file_path = "zulu-linux-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu25.28.85-ca-jdk25.0.0-linux_aarch64.tar.gz",
    )
    http_file(
        name = "openjdk_linux_ppc64le_vanilla",
        integrity = "sha256-sGC7ErOhkqBZnwPruUlUkveMSMth4pHjNqiwDneY/7A=",
        downloaded_file_path = "adoptopenjdk-ppc64le-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25%2B36/OpenJDK25U-jdk_ppc64le_linux_hotspot_25_36.tar.gz",
    )
    http_file(
        name = "openjdk_macos_x86_64_vanilla",
        integrity = "sha256-ws3h0xPZBLeTw3YCFO76IH7Mp98E58QISr3x9rvrwno=",
        downloaded_file_path = "zulu-macos-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu25.28.85-ca-jdk25.0.0-macosx_x64.tar.gz",
    )
    http_file(
        name = "openjdk_macos_aarch64_vanilla",
        integrity = "sha256-c/ZPa618PfMfunQPvLu+98Glzt7/u13zht15vHKrqbY=",
        downloaded_file_path = "zulu-macos-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu25.28.85-ca-jdk25.0.0-macosx_aarch64.tar.gz",
    )
    http_file(
        name = "openjdk_win_vanilla",
        integrity = "sha256-Xvz05qYTyuBsgEHeijaVtzRqrQMH05e2a/VSgc8aXLY=",
        downloaded_file_path = "zulu-win-vanilla.zip",
        url = "https://cdn.azul.com/zulu/bin/zulu25.28.85-ca-jdk25.0.0-win_x64.zip",
    )
    http_file(
        name = "openjdk_win_arm64_vanilla",
        integrity = "sha256-9fbYqRNpVkno4mB/4Nx5yBlTslgwE6wfuXfGPLSTW/s=",
        downloaded_file_path = "zulu-win-arm64.zip",
        url = "https://cdn.azul.com/zulu/bin/zulu25.28.85-ca-jdk25.0.0-win_aarch64.zip",
    )

    # TODO: Update to JDK 25 when available for these architectures.
    http_file(
        name = "openjdk_linux_s390x_vanilla",
        integrity = "sha256-VVBZ9JKatkNeuDtJbQuWm8appcB5FdX3YH9dgz44+zk=",
        downloaded_file_path = "adoptopenjdk-s390x-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin24-binaries/releases/download/jdk-24.0.2%2B12/OpenJDK24U-jdk_s390x_linux_hotspot_24.0.2_12.tar.gz",
    )
    http_file(
        name = "openjdk_linux_riscv64_vanilla",
        integrity = "sha256-k/ta8TSRtbBaw3gAK4qZdhTcD0IocMENX63N/LzQuUg=",
        downloaded_file_path = "adoptopenjdk-riscv64-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin24-binaries/releases/download/jdk-24.0.2%2B12/OpenJDK24U-jdk_riscv64_linux_hotspot_24.0.2_12.tar.gz",
    )

def _async_profiler_repos(ctx):
    http_file(
        name = "async_profiler",
        downloaded_file_path = "async-profiler.jar",
        # At commit f0ceda6356f05b7ad0a6593670c8c113113bf0b3 (2024-12-09).
        sha256 = "da95a5292fb203966196ecb68a39a8c26ad7276aeef642ec1de872513be1d8b3",
        urls = ["https://mirror.bazel.build/github.com/async-profiler/async-profiler/releases/download/nightly/async-profiler.jar"],
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
        sha256 = "7c6243bb91272a2797acb8cc44acf3e406e0b658a94d90d9391ca375fc961857",
        strip_prefix = "async-profiler-3.0-f0ceda6-linux-arm64/lib",
        urls = ["https://mirror.bazel.build/github.com/async-profiler/async-profiler/releases/download/nightly/async-profiler-3.0-f0ceda6-linux-arm64.tar.gz"],
    )

    http_archive(
        name = "async_profiler_linux_x64",
        build_file_content = _ASYNC_PROFILER_BUILD_TEMPLATE.format(
            ext = "so",
            tag = "linux-x64",
        ),
        sha256 = "448a3dc681375860eba2264d6cae7a848bd3f07f81f547a9ce58b742a1541d25",
        strip_prefix = "async-profiler-3.0-f0ceda6-linux-x64/lib",
        urls = ["https://mirror.bazel.build/github.com/async-profiler/async-profiler/releases/download/nightly/async-profiler-3.0-f0ceda6-linux-x64.tar.gz"],
    )

    http_archive(
        name = "async_profiler_macos",
        build_file_content = _ASYNC_PROFILER_BUILD_TEMPLATE.format(
            ext = "dylib",
            tag = "macos",
        ),
        sha256 = "0651004c78d080f67763cddde6e1f58cd0d0c4cb0b57034beef80b450ff5adf2",
        strip_prefix = "async-profiler-3.0-f0ceda6-macos/lib",
        urls = ["https://mirror.bazel.build/github.com/async-profiler/async-profiler/releases/download/nightly/async-profiler-3.0-f0ceda6-macos.zip"],
    )

# This is an extension (instead of use_repo_rule usages) only to create a
# lockfile entry for the distribution repo module extension.
async_profiler_repos = module_extension(_async_profiler_repos)
