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
)
