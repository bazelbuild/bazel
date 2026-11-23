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
load("@rules_java//toolchains:remote_java_repository.bzl", "remote_java_repository")

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
    "googleapis+",
    "googleapis-rules-registry+",
    "grpc+",
    "grpc-java+",
    "hermetic_launcher+",
    "llvm+",
    "package_metadata+",
    "platforms",
    "protobuf+",
    "protoc-gen-validate+",
    "re2+",
    "rules_android+",
    "rules_apple+",
    "rules_cc+",
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
    "rules_rs+",
    "rules_shell+",
    "rules_swift+",
    "stardoc+",
    "tar.bzl+",
    "with_cfg.bzl+",
    "xds+",
    "zlib+",
    "zstd-jni+",
]

_ASYNC_PROFILER_URL = "https://github.com/async-profiler/async-profiler/releases/download/v4.5/"

# Maps the files downloaded from _ASYNC_PROFILER_URL to their integrity.
_ASYNC_PROFILER_FILES = {
    "async-profiler.jar": "sha256-gdbK6ZcdQa8UCxxif0DVERguAEtJEwfcPH3jsZnry7s=",
    "async-profiler-4.5-linux-arm64.tar.gz": "sha256-ZMQdFGXWAJdDnFDX6SS0lG8fYrHL0hzlsDT60JwNaXk=",
    "async-profiler-4.5-linux-x64.tar.gz": "sha256-iVRvu57g/FSWx+3UCZsHCUibx4sNgFfMu0uAH2sDK2I=",
    "async-profiler-4.5-macos.zip": "sha256-RtBO+B9TKgZaCzh35IiqcGr6FKouoUQzsyPbnm/adtw=",
}

# Archives required while bootstrapping Bazel offline that are fetched by
# reproducible module extensions, as a map from URL to integrity. They are
# listed here rather than in DIST_ARCHIVE_REPOS because reproducible extensions
# are not recorded in the lockfile.
DIST_ARCHIVE_ARTIFACTS = {
    # bats-core is fetched by @bazel_lib's bats toolchain extension.
    "https://github.com/bats-core/bats-core/archive/v1.10.0.tar.gz": "sha256-oan3h1qktqlIDKOE1YZfHM8bCx+urWtHqkfXlwmlxf0=",
    # rules_rust and tinyjson are fetched by @rules_rs's reproducible
    # rules_rust extension and use_repo_rule.
    "https://github.com/hermeticbuild/rules_rust/releases/download/source-0ff2996ca9267312baab2c2c16040561a8a649bb/rules_rust-0ff2996ca9267312baab2c2c16040561a8a649bb.tar.gz": "sha256-Q55FIsJ+oVKXrjZTvQAaH2YTjIPq/3X1qoY/lSPHSLo=",
    "https://static.crates.io/crates/tinyjson/tinyjson-2.5.1.crate": "sha256-mrlXNeosj9URVNAeOc8TkSp4BxwtiavEmn7xAqfdclo=",
} | {
    _ASYNC_PROFILER_URL + file: integrity
    for file, integrity in _ASYNC_PROFILER_FILES.items()
}

##################################################################################
#
# Make sure all URLs below are mirrored to https://mirror.bazel.build
#
##################################################################################

# The JDK distributions embedded into Bazel. They are also registered as the Java runtimes
# selected by --java_runtime_version=remotejdk_27 so that tests run on the embedded JDK version.
_EMBEDDED_JDKS = {
    "openjdk_linux_vanilla": struct(
        downloaded_file_path = "zulu-linux-vanilla.tar.gz",
        integrity = "sha256-xNe/TICiHpNRh+zcsRHksvFOH90YNkl0/n99pUsSvqA=",
        remote_java_repository = "remotejdk27_linux",
        strip_prefix = "zulu27.30.201-ca-jdk27.0.1.1-linux_x64",
        target_compatible_with = ["@platforms//os:linux", "@platforms//cpu:x86_64"],
        url = "https://cdn.azul.com/zulu/bin/zulu27.30.201-ca-jdk27.0.1.1-linux_x64.tar.gz",
    ),
    "openjdk_linux_aarch64_vanilla": struct(
        downloaded_file_path = "zulu-linux-aarch64-vanilla.tar.gz",
        integrity = "sha256-GIdbkq0lNSMKExdfcYkz98wFk7Sv+i2rRCgV7CRdrLs=",
        remote_java_repository = "remotejdk27_linux_aarch64",
        strip_prefix = "zulu27.30.201-ca-jdk27.0.1.1-linux_aarch64",
        target_compatible_with = ["@platforms//os:linux", "@platforms//cpu:aarch64"],
        url = "https://cdn.azul.com/zulu/bin/zulu27.30.201-ca-jdk27.0.1.1-linux_aarch64.tar.gz",
    ),
    "openjdk_macos_x86_64_vanilla": struct(
        downloaded_file_path = "zulu-macos-vanilla.tar.gz",
        integrity = "sha256-Jx+NG7L2kr3JgXmejyURM8hUzFwe2vqAKYpGG7iQnew=",
        remote_java_repository = "remotejdk27_macos",
        strip_prefix = "zulu27.30.201-ca-jdk27.0.1.1-macosx_x64/Contents/Home",
        target_compatible_with = ["@platforms//os:macos", "@platforms//cpu:x86_64"],
        url = "https://cdn.azul.com/zulu/bin/zulu27.30.201-ca-jdk27.0.1.1-macosx_x64.tar.gz",
    ),
    "openjdk_macos_aarch64_vanilla": struct(
        downloaded_file_path = "zulu-macos-aarch64-vanilla.tar.gz",
        integrity = "sha256-HShDChPSVySIVnkuATHYRcvjJBRZcd98lxbsKpKsD8o=",
        remote_java_repository = "remotejdk27_macos_aarch64",
        strip_prefix = "zulu27.30.201-ca-jdk27.0.1.1-macosx_aarch64/Contents/Home",
        target_compatible_with = ["@platforms//os:macos", "@platforms//cpu:aarch64"],
        url = "https://cdn.azul.com/zulu/bin/zulu27.30.201-ca-jdk27.0.1.1-macosx_aarch64.tar.gz",
    ),
    "openjdk_win_vanilla": struct(
        downloaded_file_path = "zulu-win-vanilla.zip",
        integrity = "sha256-9UJOo2+DKqKKBEGlf/sd/izT2QOALAo20iV2sazEMOw=",
        remote_java_repository = "remotejdk27_win",
        strip_prefix = "zulu27.30.201-ca-jdk27.0.1.1-win_x64",
        target_compatible_with = ["@platforms//os:windows", "@platforms//cpu:x86_64"],
        url = "https://cdn.azul.com/zulu/bin/zulu27.30.201-ca-jdk27.0.1.1-win_x64.zip",
    ),
    "openjdk_linux_ppc64le_vanilla": struct(
        downloaded_file_path = "adoptopenjdk-ppc64le-vanilla.tar.gz",
        integrity = "sha256-SaNQ1uD+QffulP+vEB8JUfNQQuhsUu+op1Gjj8shJrs=",
        remote_java_repository = "remotejdk27_linux_ppc64le",
        strip_prefix = "jdk-27+35",
        target_compatible_with = ["@platforms//os:linux", "@platforms//cpu:ppc64le"],
        url = "https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B35/OpenJDK27U-jdk_ppc64le_linux_hotspot_27_35.tar.gz",
    ),
    "openjdk_linux_riscv64_vanilla": struct(
        downloaded_file_path = "adoptopenjdk-riscv64-vanilla.tar.gz",
        integrity = "sha256-Jw1013Mq26p69OFTFUEmwUTmAAu3Nv0JhUIfdQdVVao=",
        remote_java_repository = "remotejdk27_linux_riscv64",
        strip_prefix = "jdk-27+35",
        target_compatible_with = ["@platforms//os:linux", "@platforms//cpu:riscv64"],
        url = "https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B35/OpenJDK27U-jdk_riscv64_linux_hotspot_27_35.tar.gz",
    ),
    "openjdk_linux_s390x_vanilla": struct(
        downloaded_file_path = "adoptopenjdk-s390x-vanilla.tar.gz",
        integrity = "sha256-W8g6FFPCtoljIYcX8ThRLOENdVOmhDIILmVTrDH6gps=",
        remote_java_repository = "remotejdk27_linux_s390x",
        strip_prefix = "jdk-27+35",
        target_compatible_with = ["@platforms//os:linux", "@platforms//cpu:s390x"],
        url = "https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B35/OpenJDK27U-jdk_s390x_linux_hotspot_27_35.tar.gz",
    ),
    "openjdk_win_arm64_vanilla": struct(
        downloaded_file_path = "bellsoft-win-arm64.zip",
        integrity = "sha256-P0Jz+ezqm6F849dGPLVcq0RG/xzqJeNXdXTMimRJ0kU=",
        remote_java_repository = "remotejdk27_win_arm64",
        strip_prefix = "jdk-27.0.1.1",
        target_compatible_with = ["@platforms//os:windows", "@platforms//cpu:arm64"],
        # Use BellSoft Liberica for Windows ARM64 as it ships with jmods, which are
        # required for cross-jlinking the minimized JDK.
        url = "https://github.com/bell-sw/Liberica/releases/download/27.0.1.1%2B5/bellsoft-jdk27.0.1.1%2B5-windows-aarch64.zip",
    ),
}

def embedded_jdk_repositories():
    """OpenJDK distributions used to create a version of Bazel bundled with the OpenJDK."""
    for name, jdk in _EMBEDDED_JDKS.items():
        http_file(
            name = name,
            integrity = jdk.integrity,
            downloaded_file_path = jdk.downloaded_file_path,
            url = jdk.url,
        )
        remote_java_repository(
            name = jdk.remote_java_repository,
            integrity = jdk.integrity,
            strip_prefix = jdk.strip_prefix,
            target_compatible_with = jdk.target_compatible_with,
            url = jdk.url,
            version = "27",
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

    # The Windows arm64 runtime above is cross-jlinked on a Windows x64 host. Since
    # JDK 26, jlink requires the tool JDK and the target java.base to be the exact
    # same build (both vendor and build number are compared), so the jlink tool has
    # to be a Windows x64 build of the same BellSoft Liberica release as
    # openjdk_win_arm64_vanilla. It is used only as the jlink tool; the embedded
    # Windows x64 runtime itself is still Azul Zulu (openjdk_win_vanilla).
    http_file(
        name = "openjdk_win_arm64_jlink_tool",
        integrity = "sha256-hk5stmY5nqQFRHEAPDHUFDcI1xo4DQe53Zq2QMdOnek=",
        downloaded_file_path = "bellsoft-win-x64-jlink-tool.zip",
        url = "https://github.com/bell-sw/Liberica/releases/download/27.0.1.1%2B5/bellsoft-jdk27.0.1.1%2B5-windows-amd64.zip",
    )

def _async_profiler_repos(ctx):
    http_file(
        name = "async_profiler",
        downloaded_file_path = "async-profiler.jar",
        integrity = _ASYNC_PROFILER_FILES["async-profiler.jar"],
        urls = [_ASYNC_PROFILER_URL + "async-profiler.jar"],
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
        integrity = _ASYNC_PROFILER_FILES["async-profiler-4.5-linux-arm64.tar.gz"],
        strip_prefix = "async-profiler-4.5-linux-arm64/lib",
        urls = [_ASYNC_PROFILER_URL + "async-profiler-4.5-linux-arm64.tar.gz"],
    )

    http_archive(
        name = "async_profiler_linux_x64",
        build_file_content = _ASYNC_PROFILER_BUILD_TEMPLATE.format(
            ext = "so",
            tag = "linux-x64",
        ),
        integrity = _ASYNC_PROFILER_FILES["async-profiler-4.5-linux-x64.tar.gz"],
        strip_prefix = "async-profiler-4.5-linux-x64/lib",
        urls = [_ASYNC_PROFILER_URL + "async-profiler-4.5-linux-x64.tar.gz"],
    )

    http_archive(
        name = "async_profiler_macos",
        build_file_content = _ASYNC_PROFILER_BUILD_TEMPLATE.format(
            ext = "dylib",
            tag = "macos",
        ),
        integrity = _ASYNC_PROFILER_FILES["async-profiler-4.5-macos.zip"],
        strip_prefix = "async-profiler-4.5-macos/lib",
        urls = [_ASYNC_PROFILER_URL + "async-profiler-4.5-macos.zip"],
    )

    # NOTE: This is reproducible but the distdir scripts want it in the lockfile
    return ctx.extension_metadata(reproducible = False)

# This is an extension (instead of use_repo_rule usages) only to create a
# lockfile entry for the distribution repo module extension.
async_profiler_repos = module_extension(_async_profiler_repos)
