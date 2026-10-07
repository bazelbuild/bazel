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
    "bazel_lib+",
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
    "package_metadata+",
    "platforms",
    "protobuf+",
    "protoc-gen-validate+",
    "re2+",
    "rules_android+",
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
    "bats_core",
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
        integrity = "sha256-zLwVxO2+39zAPC0poqosba+eb/tM2n3V7OD8uzc1Amc=",
        downloaded_file_path = "zulu-linux-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu27.28.101-ca-jdk27.0.0-linux_x64.tar.gz",
    )
    http_file(
        name = "openjdk_linux_aarch64_vanilla",
        integrity = "sha256-n6W/hleDxDhAEB/Nw6UiK2Da9P4R8mgGJHv4B8CKujg=",
        downloaded_file_path = "zulu-linux-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu27.28.101-ca-jdk27.0.0-linux_aarch64.tar.gz",
    )
    http_file(
        name = "openjdk_linux_ppc64le_vanilla",
        integrity = "sha256-SaNQ1uD+QffulP+vEB8JUfNQQuhsUu+op1Gjj8shJrs=",
        downloaded_file_path = "adoptopenjdk-ppc64le-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B35/OpenJDK27U-jdk_ppc64le_linux_hotspot_27_35.tar.gz",
    )
    http_file(
        name = "openjdk_linux_riscv64_vanilla",
        integrity = "sha256-Jw1013Mq26p69OFTFUEmwUTmAAu3Nv0JhUIfdQdVVao=",
        downloaded_file_path = "adoptopenjdk-riscv64-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B35/OpenJDK27U-jdk_riscv64_linux_hotspot_27_35.tar.gz",
    )
    http_file(
        name = "openjdk_linux_s390x_vanilla",
        integrity = "sha256-W8g6FFPCtoljIYcX8ThRLOENdVOmhDIILmVTrDH6gps=",
        downloaded_file_path = "adoptopenjdk-s390x-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B35/OpenJDK27U-jdk_s390x_linux_hotspot_27_35.tar.gz",
    )

    # Temurin ships its JMODs separately. Keep these at the same build as the
    # corresponding JDK archives above so jlink can create the embedded runtime.
    http_file(
        name = "openjdk_linux_ppc64le_jmods",
        integrity = "sha256-YupcpqDKqWFjgcfrAqJplo/QncIwgvhAgtAZ7HdHyCY=",
        downloaded_file_path = "openjdk_linux_ppc64le_jmods.tar.gz",
        url = "https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B35/OpenJDK27U-jmods_ppc64le_linux_hotspot_27_35.tar.gz",
    )
    http_file(
        name = "openjdk_linux_riscv64_jmods",
        integrity = "sha256-ygpXfcnvyqfEOoMWLc6/VroIn/n+X65o6t6f37vvc2E=",
        downloaded_file_path = "openjdk_linux_riscv64_jmods.tar.gz",
        url = "https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B35/OpenJDK27U-jmods_riscv64_linux_hotspot_27_35.tar.gz",
    )
    http_file(
        name = "openjdk_linux_s390x_jmods",
        integrity = "sha256-Cvqr/tXnxTMbgJgwMxezMv+HhPKt0G03kuVnnpLFiQs=",
        downloaded_file_path = "openjdk_linux_s390x_jmods.tar.gz",
        url = "https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B35/OpenJDK27U-jmods_s390x_linux_hotspot_27_35.tar.gz",
    )

    http_file(
        name = "openjdk_macos_x86_64_vanilla",
        integrity = "sha256-tdZDk6IorYaA5ZNssRQkMRjT9th7Zo8CgO+MFdZ0Q5w=",
        downloaded_file_path = "zulu-macos-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu27.28.101-ca-jdk27.0.0-macosx_x64.tar.gz",
    )
    http_file(
        name = "openjdk_macos_aarch64_vanilla",
        integrity = "sha256-DY8dGRL6k469lUmeseiiqYdyX+QqoN2YSXu2BHSKjQc=",
        downloaded_file_path = "zulu-macos-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu27.28.101-ca-jdk27.0.0-macosx_aarch64.tar.gz",
    )
    http_file(
        name = "openjdk_win_vanilla",
        integrity = "sha256-mTlbZAScJ1QAV0bE9LZv24DJjDyrqhTUslp/iBXJl7g=",
        downloaded_file_path = "zulu-win-vanilla.zip",
        url = "https://cdn.azul.com/zulu/bin/zulu27.28.101-ca-jdk27.0.0-win_x64.zip",
    )
    http_file(
        name = "openjdk_win_arm64_vanilla",
        integrity = "sha256-Jgx7T/C02teSQWdCYqVEX4eiOEnRkfgaoRjx2/eGZvw=",
        downloaded_file_path = "bellsoft-win-arm64.zip",
        # Use BellSoft Liberica for Windows ARM64 as it ships with jmods, which are
        # required for cross-jlinking the minimized JDK.
        url = "https://github.com/bell-sw/Liberica/releases/download/27%2B36/bellsoft-jdk27%2B36-windows-aarch64.zip",
    )

    # The Windows arm64 runtime above is cross-jlinked on a Windows x64 host. Since
    # JDK 26, jlink requires the tool JDK and the target java.base to be the exact
    # same build (both vendor and build number are compared), so the jlink tool has
    # to be a Windows x64 build of the same BellSoft Liberica release as
    # openjdk_win_arm64_vanilla. It is used only as the jlink tool; the embedded
    # Windows x64 runtime itself is still Azul Zulu (openjdk_win_vanilla).
    http_file(
        name = "openjdk_win_arm64_jlink_tool",
        integrity = "sha256-v69LLjE9ZcpZSXTDPB2XQh0A83qOosG0DIZDrRgvnAI=",
        downloaded_file_path = "bellsoft-win-x64-jlink-tool.zip",
        url = "https://github.com/bell-sw/Liberica/releases/download/27%2B36/bellsoft-jdk27%2B36-windows-amd64.zip",
    )

def bats_core_deps():
    # These are a transitive dep of bazel_lib and marked `reproducible`, so
    # not included in the module lockfile.
    http_file(
        name = "bats_core",
        downloaded_file_path = "bats_core.tar.gz",
        integrity = "sha256-oan3h1qktqlIDKOE1YZfHM8bCx+urWtHqkfXlwmlxf0=",
        urls = ["https://github.com/bats-core/bats-core/archive/v1.10.0.tar.gz"],
    )

def _async_profiler_repos(ctx):
    http_file(
        name = "async_profiler",
        downloaded_file_path = "async-profiler.jar",
        integrity = "sha256-gdbK6ZcdQa8UCxxif0DVERguAEtJEwfcPH3jsZnry7s=",
        urls = ["https://github.com/async-profiler/async-profiler/releases/download/v4.5/async-profiler.jar"],
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
        integrity = "sha256-ZMQdFGXWAJdDnFDX6SS0lG8fYrHL0hzlsDT60JwNaXk=",
        strip_prefix = "async-profiler-4.5-linux-arm64/lib",
        urls = ["https://github.com/async-profiler/async-profiler/releases/download/v4.5/async-profiler-4.5-linux-arm64.tar.gz"],
    )

    http_archive(
        name = "async_profiler_linux_x64",
        build_file_content = _ASYNC_PROFILER_BUILD_TEMPLATE.format(
            ext = "so",
            tag = "linux-x64",
        ),
        integrity = "sha256-iVRvu57g/FSWx+3UCZsHCUibx4sNgFfMu0uAH2sDK2I=",
        strip_prefix = "async-profiler-4.5-linux-x64/lib",
        urls = ["https://github.com/async-profiler/async-profiler/releases/download/v4.5/async-profiler-4.5-linux-x64.tar.gz"],
    )

    http_archive(
        name = "async_profiler_macos",
        build_file_content = _ASYNC_PROFILER_BUILD_TEMPLATE.format(
            ext = "dylib",
            tag = "macos",
        ),
        integrity = "sha256-RtBO+B9TKgZaCzh35IiqcGr6FKouoUQzsyPbnm/adtw=",
        strip_prefix = "async-profiler-4.5-macos/lib",
        urls = ["https://github.com/async-profiler/async-profiler/releases/download/v4.5/async-profiler-4.5-macos.zip"],
    )

# This is an extension (instead of use_repo_rule usages) only to create a
# lockfile entry for the distribution repo module extension.
async_profiler_repos = module_extension(_async_profiler_repos)

def _dist_repos_impl(_ctx):
    bats_core_deps()

dist_repos = module_extension(_dist_repos_impl)
