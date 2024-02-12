# Copyright 2024 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License(**kwargs): Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing(**kwargs): software
# distributed under the License is distributed on an "AS IS" BASIS(**kwargs):
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND(**kwargs): either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Legacy apple_common module"""

load(":common/objc/linking_support.bzl", "linking_support")

native_apple_common = _builtins.internal.apple_common

apple_common = struct(
    apple_toolchain = lambda: native_apple_common.apple_toolchain(),
    platform_type = native_apple_common.platform_type,
    platform = native_apple_common.platform,
    XcodeProperties = native_apple_common.XcodeProperties,
    XcodeVersionConfig = native_apple_common.XcodeVersionConfig,
    Objc = native_apple_common.Objc,
    AppleDynamicFramework = native_apple_common.AppleDynamicFramework,
    AppleExecutableBinary = native_apple_common.AppleExecutableBinary,
    AppleDebugOutputs = native_apple_common.AppleDebugOutputs,
    apple_host_system_env = lambda xcode_config: native_apple_common.apple_host_system_env(xcode_config),
    target_apple_env = lambda *args: native_apple_common.target_apple_env(*args),
    new_objc_provider = lambda **kwargs: native_apple_common.new_objc_provider(**kwargs),
    new_dynamic_framework_provider = lambda **kwargs: native_apple_common.new_dynamic_framework_provider(**kwargs),
    new_executable_binary_provider = lambda **kwargs: native_apple_common.new_executable_binary_provider(**kwargs),
    link_multi_arch_binary = lambda **kwargs: native_apple_common.link_multi_arch_binary(**kwargs),
    link_multi_arch_static_library = linking_support.link_multi_arch_static_library,
    dotted_version = lambda version: native_apple_common.dotted_version(version),
)
