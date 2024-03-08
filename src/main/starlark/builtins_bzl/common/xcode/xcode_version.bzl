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

def _xcode_version_impl(ctx):
    xcode_version_properties = _builtins.internal.XcodeProperties(
        version = ctx.attr.version,
        default_ios_sdk_version = ctx.attr.default_ios_sdk_version,
        default_visionos_sdk_version = ctx.attr.default_visionos_sdk_version,
        default_watchos_sdk_version = ctx.attr.default_watchos_sdk_version,
        default_tvos_sdk_version = ctx.attr.default_tvos_sdk_version,
        default_macos_sdk_version = ctx.attr.default_macos_sdk_version,
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
        "aliases": attr.string_list(doc = "Accepted aliases for this version of Xcode. If the value of the xcode_version build flag matches any of the given alias strings, this xcode version will be used.", allow_empty = True, mandatory = False),
        "default_ios_sdk_version": attr.string(default = "8.4", doc = "The ios sdk version that is used by default when this version of xcode is being used. The --ios_sdk_version build flag will override the value specified here.", mandatory = False),
        "default_visionos_sdk_version": attr.string(default = "1.0", doc = "The visionos sdk version that is used by default when this version of xcode is being used. The --visionos_sdk_version build flag will override the value specified here.", mandatory = False),
        "default_watchos_sdk_version": attr.string(default = "2.0", doc = "The watchos sdk version that is used by default when this version of xcode is being used. The --watchos_sdk_version build flag will override the value specified here.", mandatory = False),
        "default_tvos_sdk_version": attr.string(default = "10.11", doc = "The tvos sdk version that is used by default when this version of xcode is being used. The --tvos_sdk_version build flag will override the value specified here.", mandatory = False),
        "default_macos_sdk_version": attr.string(default = "9.0", doc = "The macos sdk version that is used by default when this version of xcode is being used. The --macos_sdk_version build flag will override the value specified here.", mandatory = False),
    },
    implementation = _xcode_version_impl,
)
