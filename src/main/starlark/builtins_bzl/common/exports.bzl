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

"""Exported builtins symbols that are not specific to OSS Bazel."""

load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/objc/apple_common.bzl", "apple_common")
load(":common/xcode/available_xcodes.bzl", "available_xcodes")
load(":common/xcode/xcode_config.bzl", "xcode_config")
load(":common/xcode/xcode_config_alias.bzl", "xcode_config_alias")
load(":common/xcode/xcode_version.bzl", "xcode_version")

exported_toplevels = {
    # This dummy symbol is not part of the public API; it is only used to test
    # that builtins injection is working properly. Its built-in value is
    # "original value".
    "_builtins_dummy": "overridden value",
    "apple_common": apple_common,
    "proto_common_do_not_use": struct(
        INCOMPATIBLE_ENABLE_PROTO_TOOLCHAIN_RESOLUTION =
            _builtins.toplevel.proto_common_do_not_use.incompatible_enable_proto_toolchain_resolution(),
    ),
}

# A list of Starlarkified native rules.
#
# * leading `+` means the Starlark rule is used by default, but can be overridden
#   on the Bazel command line
# * no leading symbol means the Starlark rule is used and can't be overridden
# * leading `-` means the Starlark rule exists, but is not used by default
exported_rules = {
    "xcode_version": xcode_version,
    "available_xcodes": available_xcodes,
    "xcode_config": xcode_config,
    "xcode_config_alias": xcode_config_alias,
}

# A list of Starlark functions callable from native rules implementation.
exported_to_java = {
    "get_toolchain_global_make_variables": cc_helper.get_toolchain_global_make_variables,
}
