# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""Semantics for Bazel Objc rules"""

load(":common/cc/cc_common.bzl", "cc_common")

_CPU_TO_PLATFORM = {
    "darwin_x86_64": "@build_bazel_apple_support//platforms:darwin_x86_64",
    "darwin_arm64": "@build_bazel_apple_support//platforms:darwin_arm64",
    "darwin_arm64e": "@build_bazel_apple_support//platforms:darwin_arm64e",
    "ios_x86_64": "@build_bazel_apple_support//platforms:ios_x86_64",
    "ios_arm64": "@build_bazel_apple_support//platforms:ios_arm64",
    "ios_sim_arm64": "@build_bazel_apple_support//platforms:ios_sim_arm64",
    "ios_arm64e": "@build_bazel_apple_support//platforms:ios_arm64e",
    "tvos_sim_arm64": "@build_bazel_apple_support//platforms:tvos_sim_arm64",
    "tvos_arm64": "@build_bazel_apple_support//platforms:tvos_arm64",
    "tvos_x86_64": "@build_bazel_apple_support//platforms:tvos_x86_64",
    "visionos_arm64": "@build_bazel_apple_support//platforms:visionos_arm64",
    "visionos_sim_arm64": "@build_bazel_apple_support//platforms:visionos_sim_arm64",
    "visionos_x86_64": "@build_bazel_apple_support//platforms:visionos_x86_64",
    "watchos_armv7k": "@build_bazel_apple_support//platforms:watchos_armv7k",
    "watchos_arm64": "@build_bazel_apple_support//platforms:watchos_arm64",
    "watchos_device_arm64": "@build_bazel_apple_support//platforms:watchos_arm64",
    "watchos_device_arm64e": "@build_bazel_apple_support//platforms:watchos_arm64e",
    "watchos_arm64_32": "@build_bazel_apple_support//platforms:watchos_arm64_32",
    "watchos_x86_64": "@build_bazel_apple_support//platforms:watchos_x86_64",
}

def _check_toolchain_supports_objc_compile(ctx, cc_toolchain):
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        language = "objc",
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    if not cc_common.action_is_enabled(
        feature_configuration = feature_configuration,
        action_name = "objc-compile",
    ):
        fail("Compiling objc_library targets requires the Apple CC toolchain " +
             "which can be found here: https://github.com/bazelbuild/apple_support#toolchain-setup")

def _get_licenses_attr():
    # TODO(b/182226065): Change to applicable_licenses
    return {}

def _get_semantics():
    return _builtins.internal.bazel_objc_internal.semantics

def _get_repo():
    return "bazel_tools"

semantics = struct(
    check_toolchain_supports_objc_compile = _check_toolchain_supports_objc_compile,
    get_semantics = _get_semantics,
    get_repo = _get_repo,
    get_licenses_attr = _get_licenses_attr,
    cpu_to_platform = lambda cpu: _CPU_TO_PLATFORM[cpu],
)
