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

load(":common/objc/apple_platform.bzl", "PLATFORM", "PLATFORM_TYPE")

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

def _xcode_version_info_init(
        *,
        ios_sdk_version,
        ios_minimum_os_version,
        visionos_sdk_version,
        visionos_minimum_os_version,
        watchos_sdk_version,
        watchos_minimum_os_version,
        tvos_sdk_version,
        tvos_minimum_os_version,
        macos_sdk_version,
        macos_minimum_os_version,
        xcode_version,
        availability,
        xcode_version_flag,
        include_xcode_execution_info):
    execution_requirements = {
        "requires-darwin": "",
        "supports-xcode-requirements-set": "",
    }
    if availability == "LOCAL":
        execution_requirements["no-remote"] = ""
    elif availability == "REMOTE":
        execution_requirements["no-local"] = ""

    if include_xcode_execution_info:
        if xcode_version:
            execution_requirements["requires-xcode:{}".format(xcode_version)] = ""
        if xcode_version_flag:
            hyphen_index = xcode_version_flag.find("-")
            if hyphen_index != -1:
                label = xcode_version_flag[hyphen_index + 1:]
                execution_requirements["requires-xcode-label:{}".format(label)] = ""

    _apple_common = _builtins.internal.apple_common

    platform_struct = PLATFORM
    platform_type_struct = PLATFORM_TYPE

    # To preserve the original behavior (throwing an error if a version string
    # is not correctly formatted), we have to convert them to dotted versions
    # eagerly; we cannot defer it to happen inside the functions below.
    dotted_ios_minimum_os = _apple_common.dotted_version(ios_minimum_os_version)
    dotted_tvos_minimum_os = _apple_common.dotted_version(tvos_minimum_os_version)
    dotted_visionos_minimum_os = _apple_common.dotted_version(visionos_minimum_os_version)
    dotted_watchos_minimum_os = _apple_common.dotted_version(watchos_minimum_os_version)
    dotted_macos_minimum_os = _apple_common.dotted_version(macos_minimum_os_version)
    dotted_ios_sdk = _apple_common.dotted_version(ios_sdk_version)
    dotted_tvos_sdk = _apple_common.dotted_version(tvos_sdk_version)
    dotted_visionos_sdk = _apple_common.dotted_version(visionos_sdk_version)
    dotted_watchos_sdk = _apple_common.dotted_version(watchos_sdk_version)
    dotted_macos_sdk = _apple_common.dotted_version(macos_sdk_version)

    def _xcode_version(xcode_version):
        if not xcode_version:
            return None
        if xcode_version.startswith("/"):
            # Versions that represent a path on disk should be propagated as-is since they will
            # be used directly as DEVELOPER_DIR
            return xcode_version
        return _apple_common.dotted_version(xcode_version)

    def _minimum_os_for_platform_type(platform_type):
        if platform_type in (platform_type_struct.ios, platform_type_struct.catalyst):
            # Catalyst builds require usage of the iOS minimum version when
            # building, but require the usage of the macOS SDK to actually do
            # the build. This means that the particular version used for
            # Catalyst differs based on what you are using the version number
            # for - the SDK or the actual application. In this method we return
            # the OS version used for the application, and so return the iOS
            # version.
            return dotted_ios_minimum_os
        elif platform_type == platform_type_struct.tvos:
            return dotted_tvos_minimum_os
        elif platform_type == platform_type_struct.visionos:
            return dotted_visionos_minimum_os
        elif platform_type == platform_type_struct.watchos:
            return dotted_watchos_minimum_os
        elif platform_type == platform_type_struct.macos:
            return dotted_macos_minimum_os
        fail("Unhandled platform type: {}".format(platform_type))

    def _sdk_version_for_platform(platform):
        # Explicitly using the name property to checkk for equality because the platform objects
        # here may be either Starlark structs from apple_platform.bzl or Java objects from
        # ApplePlatform.java which still can come in through AppleConfiguration.java.
        # TODO(b/331163027): Consider removing the .name property from the following 5 if-statements
        # once AppleConfiguration.java is migrated, too.
        if platform.name in (platform_struct.ios_device.name, platform_struct.ios_simulator.name):
            return dotted_ios_sdk
        elif platform.name in (platform_struct.tvos_device.name, platform_struct.tvos_simulator.name):
            return dotted_tvos_sdk
        elif platform.name in (platform_struct.visionos_device.name, platform_struct.visionos_simulator.name):
            return dotted_visionos_sdk
        elif platform.name in (platform_struct.watchos_device.name, platform_struct.watchos_simulator.name):
            return dotted_watchos_sdk
        elif platform.name in (platform_struct.macos.name, platform_struct.catalyst.name):
            # Catalyst builds require usage of the iOS minimum version when
            # building, but require the usage of the macOS SDK to actually do
            # the build. This means that the particular version used for
            # Catalyst differs based on what you are using the version for. As
            # this is the SDK version specifically, we use the macOS version
            # here.
            return dotted_macos_sdk
        fail("Unhandled platform: {}".format(platform))

    # The original Java implementation of this provider declared all of these
    # APIs as functions, not fields. This is atypical for Starlark providers,
    # but the built-in Starlark provider must provide the same API.
    return {
        "xcode_version": lambda: _xcode_version(xcode_version),
        "minimum_os_for_platform_type": _minimum_os_for_platform_type,
        "sdk_version_for_platform": _sdk_version_for_platform,
        "availability": lambda: availability.lower(),
        "execution_info": lambda: execution_requirements,
    }

XcodeVersionInfo, _new_xcode_version_info = provider(
    doc = """\
The set of Apple versions computed from command line options and the
`xcode_config` rule. Note that this provider contains both the Xcode version in
use and the requested minimum OS deployment versions via the `--*os_minimum_os`
flags.
""",
    fields = {
        "xcode_version": """\
A zero-argument function that returns the `apple_common.dotted_version`
representing the Xcode version that is being used to build.

This will return `None` if no Xcode versions are available.
""",
        "minimum_os_for_platform_type": """\
A function taking a single argument that is an element of
`apple_common.platform_type`, which returns the minimum compatible OS version
for target simulator and devices for that particular platform type.
""",
        "sdk_version_for_platform": """\
A function taking a single argument that is an element of
`apple_common.platform`, which returns the version of the platform SDK that will
be used to build targets for that particular platform.
""",
        "availability": """\
A zero-argument function that returns a string representing the availability of
this Xcode version: 'remote' if the version is only available remotely, 'local'
if the version is only available locally, 'both' if the version is available
both locally and remotely, or 'unknown' if the availability could not be
determined.
""",
        "execution_info": """\
A zero-argument function that returns the execution requirements for actions
that use this Xcode configuration.
""",
    },
    init = _xcode_version_info_init,
)

def _xcode_version_properties_info_init(
        *,
        xcode_version,
        default_ios_sdk_version = "8.4",
        default_macos_sdk_version = "10.11",
        default_tvos_sdk_version = "9.0",
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
