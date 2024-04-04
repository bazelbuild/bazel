# Copyright 2024 The Bazel Authors. All rights reserved.
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

"""Rule definition for the xcode_version rule."""

load(":common/objc/apple_common.bzl", "apple_common")

def _xcode_version_impl(ctx):
    xcode_version_properties = apple_common.XcodeProperties(
        version = ctx.attr.version,
        default_ios_sdk_version = ctx.attr.default_ios_sdk_version,
        ios_sdk_minimum_os = ctx.attr.ios_sdk_minimum_os,
        default_visionos_sdk_version = ctx.attr.default_visionos_sdk_version,
        visionos_sdk_minimum_os = ctx.attr.visionos_sdk_minimum_os,
        default_watchos_sdk_version = ctx.attr.default_watchos_sdk_version,
        watchos_sdk_minimum_os = ctx.attr.watchos_sdk_minimum_os,
        default_tvos_sdk_version = ctx.attr.default_tvos_sdk_version,
        tvos_sdk_minimum_os = ctx.attr.tvos_sdk_minimum_os,
        default_macos_sdk_version = ctx.attr.default_macos_sdk_version,
        macos_sdk_minimum_os = ctx.attr.macos_sdk_minimum_os,
    )
    return [
        xcode_version_properties,
        _builtins.internal.XcodeVersionRuleData(
            label = ctx.label,
            xcode_properties = xcode_version_properties,
            aliases = ctx.attr.aliases,
        ),
        DefaultInfo(runfiles = ctx.runfiles()),
    ]

xcode_version = rule(
    attrs = {
        "version": attr.string(doc = "The official version number of a version of Xcode.", mandatory = True),
        "aliases": attr.string_list(doc = "Accepted aliases for this version of Xcode. If the value of the xcode_version build flag matches any of the given alias strings, this Xcode version will be used.", allow_empty = True, mandatory = False),
        "default_ios_sdk_version": attr.string(default = "8.4", doc = "The iOS SDK version that is used by default when this version of Xcode is being used. The `--ios_sdk_version` build flag will override the value specified here.", mandatory = False),
        "ios_sdk_minimum_os": attr.string(doc = "The minimum OS version supported by the iOS SDK for this version of Xcode.", mandatory = False),
        "default_visionos_sdk_version": attr.string(default = "1.0", doc = "The visionOS SDK version that is used by default when this version of Xcode is being used.", mandatory = False),
        "visionos_sdk_minimum_os": attr.string(doc = "The minimum OS version supported by the visionOS SDK for this version of Xcode.", mandatory = False),
        "default_watchos_sdk_version": attr.string(default = "2.0", doc = "The watchOS SDK version that is used by default when this version of Xcode is being used. The `--watchos_sdk_version` build flag will override the value specified here.", mandatory = False),
        "watchos_sdk_minimum_os": attr.string(doc = "The minimum OS version supported by the watchOS SDK for this version of Xcode.", mandatory = False),
        "default_tvos_sdk_version": attr.string(default = "10.11", doc = "The tvOS SDK version that is used by default when this version of Xcode is being used. The `--tvos_sdk_version` build flag will override the value specified here.", mandatory = False),
        "tvos_sdk_minimum_os": attr.string(doc = "The minimum OS version supported by the tvOS SDK for this version of Xcode.", mandatory = False),
        "default_macos_sdk_version": attr.string(default = "9.0", doc = "The macOS SDK version that is used by default when this version of Xcode is being used. The `--macos_sdk_version` build flag will override the value specified here.", mandatory = False),
        "macos_sdk_minimum_os": attr.string(doc = "The minimum OS version supported by the macOS SDK for this version of Xcode.", mandatory = False),
    },
    implementation = _xcode_version_impl,
)
