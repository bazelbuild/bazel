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

"""Module extensions for loading dependencies we need to build Bazel.

"""

load("@rules_graalvm//graalvm:repositories.bzl", "graalvm_repository")
load("@rules_java//toolchains:remote_java_repository.bzl", "remote_java_repository")
load("//:distdir.bzl", "repo_cache_tar")
load("//:repositories.bzl", "DIST_ARCHIVE_REPOS", "embedded_jdk_repositories")
load("//src/tools/bzlmod:utils.bzl", "parse_bazel_module_repos")
load("//tools/distributions/debian:deps.bzl", "debian_deps")

### Dependencies for building Bazel
def _bazel_build_deps(ctx):
    ctx.path(Label("//:MODULE.bazel"))  # Make sure the `bootstrap_repo_cache` repo is updated when MODULE.bazel changes.
    embedded_jdk_repositories()
    debian_deps()
    repo_cache_tar(
        name = "bootstrap_repo_cache",
        repos = DIST_ARCHIVE_REPOS,
        dirname = "derived/repository_cache",
        module_files = [
            "//:MODULE.bazel",
            "//third_party:remoteapis/MODULE.bazel",
            "//src:MODULE.tools",
        ],
    )
    BAZEL_TOOLS_DEPS_REPOS = parse_bazel_module_repos(ctx, ctx.path(Label("//src/test/tools/bzlmod:MODULE.bazel.lock")))
    repo_cache_tar(name = "bazel_tools_repo_cache", repos = BAZEL_TOOLS_DEPS_REPOS, lockfile = "//src/test/tools/bzlmod:MODULE.bazel.lock")
    graalvm_repository(
        name = "graalvm_ce",
        distribution = "ce",
        java_version = "21",
        version = "21.0.2",
    )
    graalvm_repository(
        name = "graalvm_oracle",
        distribution = "oracle",
        java_version = "21",
        version = "21.0.2",
    )
    return ctx.extension_metadata(reproducible = True)

bazel_build_deps = module_extension(implementation = _bazel_build_deps)

### JDK 27 EA repositories for testing
def _remotejdk27_repos(_ctx):
    remote_java_repository(
        name = "remotejdk27_linux",
        sha256 = "6fd828c26ea4a6614cd8ab29100a8ef00e77de1a6e71d90abc44e1a0d657914b",
        strip_prefix = "jdk-27+7",
        target_compatible_with = [
            "@platforms//os:linux",
            "@platforms//cpu:x86_64",
        ],
        urls = ["https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B7-ea-beta/OpenJDK-jdk_x64_linux_hotspot_27_7-ea.tar.gz"],
        version = "27",
    )
    remote_java_repository(
        name = "remotejdk27_linux_aarch64",
        sha256 = "319273269092d987bd53e707f889c2a5f67602256e237eaff7737f1d9939ffb4",
        strip_prefix = "jdk-27+7",
        target_compatible_with = [
            "@platforms//os:linux",
            "@platforms//cpu:aarch64",
        ],
        urls = ["https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B7-ea-beta/OpenJDK-jdk_aarch64_linux_hotspot_27_7-ea.tar.gz"],
        version = "27",
    )
    remote_java_repository(
        name = "remotejdk27_linux_ppc64le",
        sha256 = "b1e7c8d8c03547c54b1e676a3a01245ad84f3e5b8b8b975a930c8c90ccea5097",
        strip_prefix = "jdk-27+7",
        target_compatible_with = [
            "@platforms//os:linux",
            "@platforms//cpu:ppc",
        ],
        urls = ["https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B7-ea-beta/OpenJDK-jdk_ppc64le_linux_hotspot_27_7-ea.tar.gz"],
        version = "27",
    )
    remote_java_repository(
        name = "remotejdk27_linux_s390x",
        sha256 = "6a09c8a39bcab96641f06c5c4d71dabbb3fce55a81e63117448e242d24647905",
        strip_prefix = "jdk-27+7",
        target_compatible_with = [
            "@platforms//os:linux",
            "@platforms//cpu:s390x",
        ],
        urls = ["https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B7-ea-beta/OpenJDK-jdk_s390x_linux_hotspot_27_7-ea.tar.gz"],
        version = "27",
    )
    remote_java_repository(
        name = "remotejdk27_linux_riscv64",
        sha256 = "4bdd2a09124a7de1ce0cee0b1c3cc795533c1a69bfc6883554a0bc40eb071931",
        strip_prefix = "jdk-27+7",
        target_compatible_with = [
            "@platforms//os:linux",
            "@platforms//cpu:riscv64",
        ],
        urls = ["https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B7-ea-beta/OpenJDK-jdk_riscv64_linux_hotspot_27_7-ea.tar.gz"],
        version = "27",
    )
    remote_java_repository(
        name = "remotejdk27_macos",
        sha256 = "5f40bf739cd9e5b81a9b0119fdebf24ee1ac3a310437eab155225a91219ca5ef",
        strip_prefix = "jdk-27+7/Contents/Home",
        target_compatible_with = [
            "@platforms//os:macos",
            "@platforms//cpu:x86_64",
        ],
        urls = ["https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B7-ea-beta/OpenJDK-jdk_x64_mac_hotspot_27_7-ea.tar.gz"],
        version = "27",
    )
    remote_java_repository(
        name = "remotejdk27_macos_aarch64",
        sha256 = "da9765206d069168233ba56b12ebdb650697c442aaaa2035c795af0aa1e9d257",
        strip_prefix = "jdk-27+7/Contents/Home",
        target_compatible_with = [
            "@platforms//os:macos",
            "@platforms//cpu:aarch64",
        ],
        urls = ["https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B7-ea-beta/OpenJDK-jdk_aarch64_mac_hotspot_27_7-ea.tar.gz"],
        version = "27",
    )
    remote_java_repository(
        name = "remotejdk27_win",
        sha256 = "9289a73b5a6102b5bbd0cb37a88ec66685bea7516a5066ed7092593b03d8ac34",
        strip_prefix = "jdk-27+7",
        target_compatible_with = [
            "@platforms//os:windows",
            "@platforms//cpu:x86_64",
        ],
        urls = ["https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B7-ea-beta/OpenJDK-jdk_x64_windows_hotspot_27_7-ea.zip"],
        version = "27",
    )
    remote_java_repository(
        name = "remotejdk27_win_arm64",
        sha256 = "b4d1df6b90049d9ea1fc66ee872586cc382c4bb12d50c01223c765542309a06f",
        strip_prefix = "jdk-27+7",
        target_compatible_with = [
            "@platforms//os:windows",
            "@platforms//cpu:arm64",
        ],
        urls = ["https://github.com/adoptium/temurin27-binaries/releases/download/jdk-27%2B7-ea-beta/OpenJDK-jdk_aarch64_windows_hotspot_27_7-ea.zip"],
        version = "27",
    )

remotejdk27_repos = module_extension(implementation = _remotejdk27_repos)
