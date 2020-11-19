# Copyright 2017 The Bazel Authors. All rights reserved.
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

"""Deprecated: Bazel rules for creating Java toolchains. Use java_toolchain_default.bzl."""

load("@remote_java_tools_darwin//:java_toolchain_default.bzl", java_toolchain_default_macos = "java_toolchain_default")
load("@remote_java_tools_windows//:java_toolchain_default.bzl", java_toolchain_default_windows = "java_toolchain_default")
load("@remote_java_tools_linux//:java_toolchain_default.bzl", "JAVABUILDER_TOOLCHAIN_CONFIGURATION", java_toolchain_default_linux = "java_toolchain_default")

def default_java_toolchain(name, configuration = JAVABUILDER_TOOLCHAIN_CONFIGURATION, **kwargs):
    """Deprecated: Defines a remote java_toolchain with appropriate defaults for Bazel.

    Use use java_toolchain_default instead."""

    java_toolchain_default_macos(name + "_darwin", configuration = configuration, **kwargs)
    java_toolchain_default_windows(name + "_windows", configuration = configuration, **kwargs)
    java_toolchain_default_linux(name + "_linux", configuration = configuration, **kwargs)

    native.alias(
        name = name,
        actual = select({
            "@bazel_tools//src/conditions:linux_x86_64": name + "_linux",
            "@bazel_tools//src/conditions:darwin": name + "_darwin",
            "@bazel_tools//src/conditions:windows": name + "_windows",
            # On different platforms the linux repository can be used.
            # The deploy jars inside the linux repository are platform-agnostic.
            # The ijar target inside the repository identifies the different
            # platform and builds ijar from source instead of returning the
            # precompiled binary.
            "//conditions:default": name + "_linux",
        }),
    )
