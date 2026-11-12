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

"""
A variant of `native_image` that uses the auto-configured C++ toolchain when
targeting Windows.

GraalVM Native Image requires Visual Studio on Windows and fails with the MinGW
based hermetic LLVM toolchains that .bazelrc enables by default. Toolchains
that are listed later in `--extra_toolchains` take precedence, so this adds the
auto-configured toolchains after the LLVM ones.
"""

load("@rules_graalvm//graalvm:defs.bzl", _native_image = "native_image")
load("@with_cfg.bzl", "with_cfg")

native_image, _native_image_internal = (
    with_cfg(_native_image)
        .extend("extra_toolchains", select({
        Label("@platforms//os:windows"): ["@local_config_cc_toolchains//:all"],
        "//conditions:default": [],
    }))
        .build()
)
