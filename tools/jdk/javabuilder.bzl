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

    Note: Custom plugins run inside the isolated Javac classloader environment.
    To avoid ClassNotFoundException due to Javac's classloader masking:
      1. Any custom BugChecker classes must be listed in a service provider file:
         META-INF/services/com.google.errorprone.bugpatterns.BugChecker
      2. The masking classloader automatically whitelists and delegates any classes
         whose package name starts with the package prefix of the discovered checkers
         (e.g., if a checker class is `com.example.MyChecker`, classes in `com.example.*`
         will be loaded successfully).
      3. If your plugin depends on separate third-party libraries (e.g. org.json.*),
         you must relocate (shade) those dependency classes under the checker's package
         prefix (e.g. relocating to `com.example.shaded.org.json.*`). A common tool to do
         this in Bazel is `rules_jarjar` (https://github.com/bazelbuild/rules_jarjar) using
         its relocation rules.

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
