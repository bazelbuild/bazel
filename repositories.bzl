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

# Protobuf's prebuilt protoc repositories are reproducible custom repository
# rules, so their downloads are not recorded in MODULE.bazel.lock. Keep these
# in sync with @com_google_protobuf//bazel/private/oss/toolchains/prebuilt:
# tool_integrity.bzl.
PROTOBUF_PREBUILT_PROTOC_ARTIFACTS = {
    "https://github.com/protocolbuffers/protobuf/releases/download/v36.0/{}".format(filename): sha256
    for filename, sha256 in {
        "protoc-36.0-linux-aarch_64.zip": "4a00ec5e256d20a3deadd9e77d56da0ac04c72367c3c959f6d08e110a368400a",
        "protoc-36.0-linux-ppcle_64.zip": "9bd9154c503928c8a48b6b5dee8c82881c650d753f644af75e1156ebab0c1df2",
        "protoc-36.0-linux-s390_64.zip": "27556b282e33d6e04300d05ed4c38bf85e5a92d7080efedd5c41fd02cf9e3f5b",
        "protoc-36.0-linux-x86_32.zip": "60fc1d382cbbad06584761e1bf1a79e4eeefa33a8667aad5a4184e4d31c45ddb",
        "protoc-36.0-linux-x86_64.zip": "bc8211ce760bd43ee21ddc145d6d9dbaeeabae205267a79d9054a240e367d4b4",
        "protoc-36.0-osx-aarch_64.zip": "b6bc4afdcb880124bf342851d05155b6e3d9b6e661236d87b9c614250d26ae00",
        "protoc-36.0-osx-x86_64.zip": "2847d952ecd1c466769ae3ca319c9cd34c3613542eba335dc9b02c49537f6c70",
        "protoc-36.0-win32.zip": "50b904fe09980eb44d566755eb65b4693a0deac30998ea3323c6901d9540e60d",
        "protoc-36.0-win64.zip": "510fb2369a4720adb457783768c5e1481a0e1137d9e7694478dfe9b8fe445dc1",
    }.items()
}

##################################################################################
#
# Make sure all URLs below are mirrored to https://mirror.bazel.build
#
##################################################################################

def embedded_jdk_repositories():
    """OpenJDK distributions used to create a version of Bazel bundled with the OpenJDK."""
    http_file(
        name = "openjdk_linux_vanilla",
        integrity = "sha256-fWZj6o1CmN9l3gZeMvn0SXRf9gfTC6XRN3fLkunUYT0=",
        downloaded_file_path = "zulu-linux-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu26.30.11-ca-jdk26.0.1-linux_x64.tar.gz",
    )
    http_file(
        name = "openjdk_linux_aarch64_vanilla",
        integrity = "sha256-zBtFncRC10IrRqO1/lKsrqVIefp5E+KaBWUM71Rof18=",
        downloaded_file_path = "zulu-linux-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu26.30.11-ca-jdk26.0.1-linux_aarch64.tar.gz",
    )
    http_file(
        name = "openjdk_linux_ppc64le_vanilla",
        integrity = "sha256-YOAW+vQXeEBDADXZSPg/KIfVVv5RK3jB1DsyAyL+ZoU=",
        downloaded_file_path = "adoptopenjdk-ppc64le-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin26-binaries/releases/download/jdk-26.0.1%2B8/OpenJDK26U-jdk_ppc64le_linux_hotspot_26.0.1_8.tar.gz",
    )
    http_file(
        name = "openjdk_linux_riscv64_vanilla",
        integrity = "sha256-8bdi1thlmWJ5g98gDyFbyXBESmlxWco/rpMgh1a0RxU=",
        downloaded_file_path = "adoptopenjdk-riscv64-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin26-binaries/releases/download/jdk-26.0.1%2B8/OpenJDK26U-jdk_riscv64_linux_hotspot_26.0.1_8.tar.gz",
    )
    http_file(
        name = "openjdk_linux_s390x_vanilla",
        integrity = "sha256-lC3n3tFCdZKipLbb6kCDstCJHeJibHhj6XDePigZqT8=",
        downloaded_file_path = "adoptopenjdk-s390x-vanilla.tar.gz",
        url = "https://github.com/adoptium/temurin26-binaries/releases/download/jdk-26.0.1%2B8/OpenJDK26U-jdk_s390x_linux_hotspot_26.0.1_8.tar.gz",
    )
    http_file(
        name = "openjdk_macos_x86_64_vanilla",
        integrity = "sha256-GSYQQQ3N+27cokKbDV0rHN8yL1MiGKwAMn4bz43+KbM=",
        downloaded_file_path = "zulu-macos-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu26.30.11-ca-jdk26.0.1-macosx_x64.tar.gz",
    )
    http_file(
        name = "openjdk_macos_aarch64_vanilla",
        integrity = "sha256-fxsSMjJTejCm7UqofWqE1Ca3Wr81Dr4jVnhKJh6dYHY=",
        downloaded_file_path = "zulu-macos-aarch64-vanilla.tar.gz",
        url = "https://cdn.azul.com/zulu/bin/zulu26.30.11-ca-jdk26.0.1-macosx_aarch64.tar.gz",
    )
    http_file(
        name = "openjdk_win_vanilla",
        integrity = "sha256-j3b0CLDiKJdLDJV4qSdZGJu13Pp/elIVgnd+QEoyRKA=",
        downloaded_file_path = "zulu-win-vanilla.zip",
        url = "https://cdn.azul.com/zulu/bin/zulu26.30.11-ca-jdk26.0.1-win_x64.zip",
    )
    http_file(
        name = "openjdk_win_arm64_vanilla",
        integrity = "sha256-JMBoQdovyQStuciKVAawKJZmXkxqKpGYFyO0jiVUzPM=",
        downloaded_file_path = "bellsoft-win-arm64.zip",
        # BellSoft Liberica is currently the only vendor with a GA JDK 26 build
        # for Windows ARM64. It ships with jmods, which are required for
        # cross-jlinking the minimized JDK.
        url = "https://github.com/bell-sw/Liberica/releases/download/26.0.1%2B10/bellsoft-jdk26.0.1%2B10-windows-aarch64.zip",
    )

    # The Windows arm64 runtime above is cross-jlinked on a Windows x64 host. Since
    # JDK 26, jlink requires the tool JDK and the target java.base to be the exact
    # same build (both vendor and build number are compared), so the jlink tool has
    # to be a Windows x64 build of the same BellSoft Liberica release as
    # openjdk_win_arm64_vanilla. It is used only as the jlink tool; the embedded
    # Windows x64 runtime itself is still Azul Zulu (openjdk_win_vanilla).
    http_file(
        name = "openjdk_win_arm64_jlink_tool",
        integrity = "sha256-En7H6N+8rEOUfa+NziEm1slGG92cG10N9V9aoW1v48s=",
        downloaded_file_path = "bellsoft-win-x64-jlink-tool.zip",
        url = "https://github.com/bell-sw/Liberica/releases/download/26.0.1%2B10/bellsoft-jdk26.0.1%2B10-windows-amd64.zip",
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
        integrity = "sha256-hwOrB7gKRnaucBvdJPD/PMONf0OuJEHOlXtFFyOFh+c=",
        urls = ["https://github.com/async-profiler/async-profiler/releases/download/v4.4/async-profiler.jar"],
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
        integrity = "sha256-hv+XtENqzNtte7ZcHPbjinVvIDepIZlNj6HcuX0dxTw=",
        strip_prefix = "async-profiler-4.4-linux-arm64/lib",
        urls = ["https://github.com/async-profiler/async-profiler/releases/download/v4.4/async-profiler-4.4-linux-arm64.tar.gz"],
    )

    http_archive(
        name = "async_profiler_linux_x64",
        build_file_content = _ASYNC_PROFILER_BUILD_TEMPLATE.format(
            ext = "so",
            tag = "linux-x64",
        ),
        integrity = "sha256-EjPyb8lXU+dc4yczu8r48L7cLAmLDnmK+Hk1sIpjsk4=",
        strip_prefix = "async-profiler-4.4-linux-x64/lib",
        urls = ["https://github.com/async-profiler/async-profiler/releases/download/v4.4/async-profiler-4.4-linux-x64.tar.gz"],
    )

    http_archive(
        name = "async_profiler_macos",
        build_file_content = _ASYNC_PROFILER_BUILD_TEMPLATE.format(
            ext = "dylib",
            tag = "macos",
        ),
        integrity = "sha256-YXfr5W0IjRFuG0NmGPGLMxa55BiF/nQ1Ofa8KXpIcjk=",
        strip_prefix = "async-profiler-4.4-macos/lib",
        urls = ["https://github.com/async-profiler/async-profiler/releases/download/v4.4/async-profiler-4.4-macos.zip"],
    )

# This is an extension (instead of use_repo_rule usages) only to create a
# lockfile entry for the distribution repo module extension.
async_profiler_repos = module_extension(_async_profiler_repos)

def _dist_repos_impl(_ctx):
    bats_core_deps()

dist_repos = module_extension(_dist_repos_impl)
