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
    "chicory+",
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
        integrity = "sha256-Kf6gF8A8ZFIhujEgjlENeuSPVzW6QWnVZcRst35/ZvI=",
        downloaded_file_path = "zulu-linux-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu24.28.83-ca-jdk24.0.0-linux_x64.tar.gz",
    )
    http_file(
        name = "openjdk_linux_aarch64_vanilla",
        integrity = "sha256-6J7szd/ax9xCMNA9efw9Bhgv/VwQFXz5glWIoj+UYIc=",
        downloaded_file_path = "zulu-linux-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu24.28.83-ca-jdk24.0.0-linux_aarch64.tar.gz",
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
        name = "openjdk_linux_riscv64_vanilla",
        integrity = "sha256-gNe6uflhS9+TTGvEQQMb0f6tOuqF8WdwEjvYprzcUrY=",
        downloaded_file_path = "adoptopenjdk-riscv64-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin23-binaries/releases/download/jdk-23.0.1%2B11/OpenJDK23U-jdk_riscv64_linux_hotspot_23.0.1_11.tar.gz",
    )
    http_file(
        name = "openjdk_macos_x86_64_vanilla",
        integrity = "sha256-e7KJtJ9+mFFSdKCj68thfTXguWH5zXaSSb9phzXf/lQ=",
        downloaded_file_path = "zulu-macos-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu24.28.83-ca-jdk24.0.0-macosx_x64.tar.gz",
    )
    http_file(
        name = "openjdk_macos_aarch64_vanilla",
        integrity = "sha256-7yXLOJCK0RZ8V1vsexOGxGR9NAwi/pCl95BlO8E8nGU=",
        downloaded_file_path = "zulu-macos-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu24.28.83-ca-jdk24.0.0-macosx_aarch64.tar.gz",
    )
    http_file(
        name = "openjdk_win_vanilla",
        integrity = "sha256-Nfmnb2gAmoKWgefl801WVjTNxxaaT+TmbwSzJ8uccf8=",
        downloaded_file_path = "zulu-win-vanilla.zip",
        url = "https://cdn.azul.com/zulu/bin/zulu24.28.83-ca-jdk24.0.0-win_x64.zip",
    )

    # Later version of the JDK for Windows ARM64 are not available yet.
    http_file(
        name = "openjdk_win_arm64_vanilla",
        integrity = "sha256-V8VoNVuX0ojxK3IHYNgCsaGcVemwcHpcKtdtNP2JPbg=",
        downloaded_file_path = "zulu-win-arm64.zip",
        url = "https://cdn.azul.com/zulu/bin/zulu21.40.17-ca-jdk21.0.6-win_aarch64.zip",
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
