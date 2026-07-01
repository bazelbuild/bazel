# Copyright 2026 The Bazel Authors. All rights reserved.
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

"""Macros to customize JavaBuilder and customize Error Prone plugins."""

load("@rules_java//java:defs.bzl", "java_binary", "java_library")

def errorprone_with_custom_plugins(name, errorprone, plugins = [], **kwargs):
    """Merges a base Error Prone compiler core library with extra custom plugin/checker libraries.

    Args:
      name: The name of the target.
      errorprone: The base Error Prone core target.
      plugins: The list of custom checker plugins.
      **kwargs: Additional attributes to pass to java_library.
    """
    java_library(
        name = name,
        exports = [
            errorprone,
            "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:errorprone",
        ] + plugins,
        **kwargs
    )

def default_javabuilder(name, errorprone = [], javabuilder_core = "@bazel_tools//tools/jdk:javabuilder_core", **kwargs):
    """Builds a custom JavaBuilder deployment jar.

    Args:
      name: The name of the target.
      errorprone: The customized Error Prone target.
      javabuilder_core: The core JavaBuilder libraries target.
      **kwargs: Additional attributes to pass to java_binary.
    """
    if "deps" in kwargs:
        fail("deps attribute is not supported in default_javabuilder, use errorprone or javabuilder_core instead")

    java_binary(
        name = name,
        main_class = "com.google.devtools.build.buildjar.BazelJavaBuilder",
        runtime_deps = [
            javabuilder_core,
        ] + errorprone,
        **kwargs
    )
