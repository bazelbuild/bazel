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

load("//:distdir.bzl", "dist_http_file")

def embedded_jdk_repositories():
    """OpenJDK distributions used to create a version of Bazel bundled with the OpenJDK."""
    dist_http_file(
        name = "openjdk_linux_vanilla",
        downloaded_file_path = "zulu-linux-vanilla.tar.gz",
    )

    dist_http_file(
        name = "openjdk_linux_aarch64_vanilla",
        downloaded_file_path = "zulu-linux-aarch64-vanilla.tar.gz",
    )

    dist_http_file(
        name = "openjdk_linux_ppc64le_vanilla",
        downloaded_file_path = "adoptopenjdk-ppc64le-vanilla.tar.gz",
    )

    dist_http_file(
        name = "openjdk_linux_s390x_vanilla",
        downloaded_file_path = "adoptopenjdk-s390x-vanilla.tar.gz",
    )

    dist_http_file(
        name = "openjdk_macos_x86_64_vanilla",
        downloaded_file_path = "zulu-macos-vanilla.tar.gz",
    )

    dist_http_file(
        name = "openjdk_macos_aarch64_vanilla",
        downloaded_file_path = "zulu-macos-aarch64-vanilla.tar.gz",
    )

    dist_http_file(
        name = "openjdk_win_vanilla",
        downloaded_file_path = "zulu-win-vanilla.zip",
    )

    dist_http_file(
        name = "openjdk_win_arm64_vanilla",
        downloaded_file_path = "zulu-win-arm64.zip",
    )
