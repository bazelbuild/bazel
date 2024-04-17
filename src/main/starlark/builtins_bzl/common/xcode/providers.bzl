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

"""Definitions of providers used by the Xcode rules and their clients."""

AvailableXcodesInfo = provider(
    doc = """\
The available Xcode versions computed from the `available_xcodes` rule.
""",
    fields = {
        "available_versions": """\
The available Xcode versions from `available_xcodes`.
""",
        "default_version": """\
The default Xcode version from `available_xcodes`.
""",
    },
)

def _xcode_version_properties_info_init(
        xcode_version,
        default_ios_sdk_version = "8.4",
        default_macos_sdk_version = "9.0",
        default_tvos_sdk_version = "10.11",
        default_watchos_sdk_version = "2.0",
        default_visionos_sdk_version = "1.0"):
    # Ensure that all fields get default values if they weren't specified.
    return {
        "xcode_version": xcode_version,
        "default_ios_sdk_version": default_ios_sdk_version,
        "default_macos_sdk_version": default_macos_sdk_version,
        "default_tvos_sdk_version": default_tvos_sdk_version,
        "default_watchos_sdk_version": default_watchos_sdk_version,
        "default_visionos_sdk_version": default_visionos_sdk_version,
    }

XcodeVersionPropertiesInfo, _new_xcode_version_properties_info = provider(
    doc = """\
Information about a specific Xcode version, such as its default SDK versions.
""",
    fields = {
        "xcode_version": """\
A string representing the Xcode version number, or `None` if it is unknown.
""",
        "default_ios_sdk_version": """\
A string representing the default iOS SDK version number for this version of
Xcode, or `None` if it is unknown.
""",
        "default_macos_sdk_version": """\
A string representing the default macOS SDK version number for this version of
Xcode, or `None` if it is unknown.
""",
        "default_tvos_sdk_version": """\
A string representing the default tvOS SDK version number for this version of
Xcode, or `None` if it is unknown.
""",
        "default_watchos_sdk_version": """\
A string representing the default watchOS SDK version number for this version of
Xcode, or `None` if it is unknown.
""",
        "default_visionos_sdk_version": """\
A string representing the default visionOS SDK version number for this version
of Xcode, or `None` if it is unknown.
""",
    },
    init = _xcode_version_properties_info_init,
)

XcodeVersionRuleInfo = provider(
    doc = """\
The information in a single target of the `xcode_version` rule. A single target
of this rule contains an official version label decided by Apple, a number of
supported aliases one might use to reference this version, and various
properties of the Xcode version (such as default SDK versions).

For example, one may want to reference official Xcode version 7.0.1 using the
"7" or "7.0" aliases. This official version of Xcode may have a default
supported iOS SDK of 9.0.
""",
    fields = {
        "aliases": """\
A list of strings denoting aliases that can be used to reference this Xcode
version.
""",
        "label": """\
The build `Label` of the `xcode_version` target that propagated this provider.
""",
        "xcode_version_properties": """\
An `XcodeVersionPropertiesInfo` provider that contains the details about this
Xcode version, such as its default SDK versions.
""",
    },
)
