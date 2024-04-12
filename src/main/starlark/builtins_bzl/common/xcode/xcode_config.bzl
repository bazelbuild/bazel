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

"""Rule definition for the xcode_config rule."""

load(":common/objc/apple_common.bzl", "apple_common")
load(":common/xcode/semantics.bzl", "unavailable_xcode_message")

def _xcode_config_impl(ctx):
    apple_fragment = ctx.fragments.apple
    cpp_fragment = ctx.fragments.cpp

    explicit_default_version = ctx.attr.default[_builtins.internal.XcodeVersionRuleData] if ctx.attr.default else None
    explicit_versions = [
        target[_builtins.internal.XcodeVersionRuleData]
        for target in ctx.attr.versions
    ] if ctx.attr.versions else []
    remote_versions = [
        target
        for target in ctx.attr.remote_versions[_builtins.internal.AvailableXcodesInfo].available_versions
    ] if ctx.attr.remote_versions else []
    local_versions = [
        target
        for target in ctx.attr.local_versions[_builtins.internal.AvailableXcodesInfo].available_versions
    ] if ctx.attr.local_versions else []

    local_default_version = ctx.attr.local_versions[_builtins.internal.AvailableXcodesInfo].default_version if ctx.attr.local_versions else None
    xcode_version_properties = None
    availability = "unknown"

    if _use_available_xcodes(
        explicit_default_version,
        explicit_versions,
        local_versions,
        remote_versions,
    ):
        xcode_version_properties, availability = _resolve_xcode_from_local_and_remote(
            local_versions,
            remote_versions,
            apple_fragment.xcode_version_flag,
            apple_fragment.prefer_mutual_xcode,
            local_default_version,
        )
    else:
        xcode_version_properties = _resolve_explicitly_defined_version(
            explicit_versions,
            explicit_default_version,
            apple_fragment.xcode_version_flag,
        )
        availability = "UNKNOWN"

    ios_sdk_version = apple_fragment.ios_sdk_version_flag or _dotted_version_or_default(xcode_version_properties.default_ios_sdk_version, "8.4")
    macos_sdk_version = apple_fragment.macos_sdk_version_flag or _dotted_version_or_default(xcode_version_properties.default_macos_sdk_version, "10.11")
    tvos_sdk_version = apple_fragment.tvos_sdk_version_flag or _dotted_version_or_default(xcode_version_properties.default_tvos_sdk_version, "9.0")
    watchos_sdk_version = apple_fragment.watchos_sdk_version_flag or _dotted_version_or_default(xcode_version_properties.default_watchos_sdk_version, "2.0")
    visionos_sdk_version = _dotted_version_or_default(xcode_version_properties.default_visionos_sdk_version, "1.0")

    ios_minimum_os = apple_fragment.ios_minimum_os_flag or ios_sdk_version
    macos_minimum_os = apple_fragment.macos_minimum_os_flag or macos_sdk_version
    tvos_minimum_os = apple_fragment.tvos_minimum_os_flag or tvos_sdk_version
    watchos_minimum_os = apple_fragment.watchos_minimum_os_flag or watchos_sdk_version
    if cpp_fragment.minimum_os_version():
        visionos_minimum_os = _builtins.internal.apple_common.dotted_version(cpp_fragment.minimum_os_version())
    else:
        visionos_minimum_os = visionos_sdk_version

    xcode_versions = _builtins.internal.XcodeConfigInfo(
        iosMinimumOsVersion = str(ios_minimum_os),
        visionosSdkVersion = str(visionos_sdk_version),
        visionosMinimumOsVersion = str(visionos_minimum_os),
        watchosSdkVersion = str(watchos_sdk_version),
        watchosMinimumOsVersion = str(watchos_minimum_os),
        iosSdkVersion = str(ios_sdk_version),
        tvosSdkVersion = str(tvos_sdk_version),
        tvosMinimumOsVersion = str(tvos_minimum_os),
        macosSdkVersion = str(macos_sdk_version),
        macosMinimumOsVersion = str(macos_minimum_os),
        xcodeVersion = xcode_version_properties.xcode_version,
        availability = availability,
        xcodeVersionFlag = apple_fragment.xcode_version_flag,
        includeXcodeExecutionInfo = apple_fragment.include_xcode_exec_requirements,
    )
    return [
        DefaultInfo(runfiles = ctx.runfiles()),
        xcode_versions,
        xcode_version_properties,
    ]

xcode_config = rule(
    attrs = {
        "default": attr.label(
            doc = """\
The default official version of Xcode to use.
The version specified by the provided `xcode_version` target is to be used if
no `xcode_version` build flag is specified. This is required if any
`versions` are set. This may not be set if `remote_versions` or
`local_versions` is set.
""",
            flags = ["NONCONFIGURABLE"],
            providers = [[_builtins.internal.XcodeVersionRuleData]],
        ),
        "versions": attr.label_list(
            doc = """\
Accepted `xcode_version` targets that may be used.
If the value of the `xcode_version` build flag matches one of the aliases
or version number of any of the given `xcode_version` targets, the matching
target will be used. This may not be set if `remote_versions` or
`local_versions` is set.
""",
            flags = ["NONCONFIGURABLE"],
            providers = [[_builtins.internal.XcodeVersionRuleData]],
        ),
        "remote_versions": attr.label(
            doc = """\
The `xcode_version` targets that are available remotely.
These are used along with `remote_versions` to select a mutually available
version. This may not be set if `versions` is set.
""",
            flags = ["NONCONFIGURABLE"],
            providers = [[_builtins.internal.AvailableXcodesInfo]],
        ),
        "local_versions": attr.label(
            doc = """\
The `xcode_version` targets that are available locally.
These are used along with `local_versions` to select a mutually available
version. This may not be set if `versions` is set.
""",
            flags = ["NONCONFIGURABLE"],
            providers = [[_builtins.internal.AvailableXcodesInfo]],
        ),
    },
    doc = """\
A single target of this rule can be referenced by the `--xcode_version_config`
build flag to translate the `--xcode_version` flag into an accepted official
Xcode version. This allows selection of an official Xcode version from a number
of registered aliases.
""",
    fragments = ["apple", "cpp"],
    implementation = _xcode_config_impl,
)

def _use_available_xcodes(explicit_default_version, explicit_versions, local_versions, remote_versions):
    if remote_versions:
        if explicit_versions:
            fail("'versions' may not be set if '[local,remote]_versions' is set.")
        if explicit_default_version:
            fail("'default' may not be set if '[local,remote]_versions' is set.")
        if not local_versions:
            fail("if 'remote_versions' are set, you must also set 'local_versions'")
        return True
    return False

def _duplicate_alias_error(alias, versions):
    labels_containing_alias = []
    for version in versions:
        if alias in version.aliases or (str(version.version) == alias):
            labels_containing_alias.append(str(version.label))
    return "'{}' is registered to multiple labels ({}) in a single xcode_config rule".format(
        alias,
        ", ".join(labels_containing_alias),
    )

def _aliases_to_xcode_version(versions):
    version_map = {}
    if not versions:
        return version_map
    for version in versions:
        for alias in version.aliases:
            if alias in version_map:
                fail(_duplicate_alias_error(alias, versions))
            else:
                version_map[alias] = version
        if str(version.version) not in version.aliases:  # only add the version if it's not also an alias
            if str(version.version) in version_map:
                fail(_duplicate_alias_error(str(version.version), versions))
            else:
                version_map[str(version.version)] = version
    return version_map

def _resolve_xcode_from_local_and_remote(
        local_versions,
        remote_versions,
        xcode_version_flag,
        prefer_mutual_xcode,
        local_default_version):
    local_alias_to_version_map = _aliases_to_xcode_version(local_versions)
    remote_alias_to_version_map = _aliases_to_xcode_version(remote_versions)

    # A version is mutually available (available both locally and remotely) if the local version
    # attribute matches either the version attribute or one of the aliases of the remote version.
    # mutually_vailable_versions is a subset of remote_versions.
    # We assume the "version" attribute in local xcode_version contains a full version string,
    # e.g. including the build, while the versions in "alias" attribute may be less granular.
    # We don't make this assumption for remote xcode_versions.
    mutually_available_versions = {}
    for version in local_versions:
        if str(version.version) in remote_alias_to_version_map:
            mutually_available_versions[str(version.version)] = remote_alias_to_version_map[str(version.version)]

    # We'd log an event here if we could!!
    if xcode_version_flag:
        remote_version_from_flag = remote_alias_to_version_map.get(xcode_version_flag)
        local_version_from_flag = local_alias_to_version_map.get(xcode_version_flag)
        availability = "BOTH"

        if remote_version_from_flag and local_version_from_flag:
            local_version_from_remote_versions = remote_alias_to_version_map.get(str(local_version_from_flag.version))
            if local_version_from_remote_versions:
                return remote_version_from_flag.xcode_version_properties, availability
            else:
                fail(
                    ("Xcode version {0} was selected, either because --xcode_version was passed, or" +
                     " because xcode-select points to this version locally. This corresponds to" +
                     " local Xcode version {1}. That build of version {0} is not available" +
                     " remotely, but there is a different build of version {2}, which has" +
                     " version {2} and aliases {3}. You probably meant to use this version." +
                     " Please download it *and delete version {1}, then run `blaze shutdown`" +
                     " to continue using dynamic execution. If you really did intend to use" +
                     " local version {1}, please specify it fully with --xcode_version={1}.").format(
                        xcode_version_flag,
                        local_version_from_flag.version,
                        remote_version_from_flag.version,
                        remote_version_from_flag.aliases,
                    ),
                )

        elif local_version_from_flag:
            error = (
                " --xcode_version={} specified, but it is not available remotely. Actions " +
                "requiring Xcode will be run locally, which could make your build slower."
            ).format(
                xcode_version_flag,
            )
            if (mutually_available_versions):
                error += " Consider using one of [{}].".format(
                    ", ".join([version for version in mutually_available_versions]),
                )
            print(error)
            return local_version_from_flag.xcode_version_properties, "LOCAL"

        elif remote_version_from_flag:
            print(("--xcode_version={version} specified, but it is not available locally. " +
                   "Your build will fail if any actions require a local Xcode. " +
                   "If you believe you have '{version}' installed, try running {command}," +
                   "and then re-run your command. Locally available versions: {local_versions}. ")
                .format(
                version = xcode_version_flag,
                command = unavailable_xcode_message,
                local_versions = ", ".join([version for version in local_alias_to_version_map.keys()]),
            ))
            availability = "REMOTE"

            return remote_version_from_flag.xcode_version_properties, availability

        else:  # fail if we can't find any version to match
            fail(
                ("--xcode_version={0} specified, but '{0}' is not an available Xcode version." +
                 " Locally available versions: [{2}]. Remotely available versions: [{3}]. If" +
                 " you believe you have '{0}' installed, try running {1}, and then" +
                 " re-run your command.").format(
                    xcode_version_flag,
                    unavailable_xcode_message,
                    ", ".join([str(version.version) for version in local_versions]),
                    ", ".join([str(version.version) for version in remote_versions]),
                ),
            )

    # --xcode_version is not set
    availability = "UNKNOWN"
    local_version = None

    # If there aren't any mutually available versions, select the local default.
    if not mutually_available_versions:
        print(
            ("Using a local Xcode version, '{}', since there are no" +
             " remotely available Xcodes on this machine. Consider downloading one of the" +
             " remotely available Xcode versions ({}) in order to get the best build" +
             " performance.").format(local_default_version.version, ", ".join([str(version.version) for version in remote_versions])),
        )
        local_version = local_default_version
        availability = "LOCAL"
    elif (str(local_default_version.version) in remote_alias_to_version_map):
        #  If the local default version is also available remotely, use it.
        availability = "BOTH"
        local_version = remote_alias_to_version_map.get(str(local_default_version.version))
    else:
        # If an alias of the local default version is available remotely, use it.
        for version_number in local_default_version.aliases:
            if version_number in remote_alias_to_version_map:
                availability = "BOTH"
                local_version = remote_alias_to_version_map.get(str(version_number))
                break

    if local_version:
        return local_version.xcode_version_properties, availability

    # The local default is not available remotely.
    if prefer_mutual_xcode:
        # If we prefer a mutually available version, the newest one.
        newest_version = _builtins.internal.apple_common.dotted_version("0.0")
        default_version = None
        for _, version in mutually_available_versions.items():
            if version.version.compare_to(newest_version) > 0:
                default_version = version
                newest_version = default_version.version

        return default_version.xcode_version_properties, "BOTH"
    else:
        # Use the local default
        return local_default_version.xcode_version_properties, "LOCAL"

def _resolve_explicitly_defined_version(
        explicit_versions,
        explicit_default_version,
        xcode_version_flag):
    if explicit_default_version and explicit_default_version.label not in [
        version.label
        for version in explicit_versions
    ]:
        fail(
            "default label '{}' must be contained in versions attribute".format(explicit_default_version.label),
        )
    if not explicit_versions:
        if explicit_default_version:
            fail("default label must be contained in versions attribute")
        return apple_common.XcodeProperties(version = None)

    if not explicit_default_version:
        fail("if any versions are specified, a default version must be specified")

    alias_to_versions = _aliases_to_xcode_version(explicit_versions)
    if xcode_version_flag:
        flag_version = alias_to_versions.get(str(xcode_version_flag))
        if flag_version:
            return flag_version.xcode_version_properties
        else:
            fail(
                ("--xcode_version={0} specified, but '{0}' is not an available Xcode version. " +
                 "If you believe you have '{0}' installed, try running \"bazel shutdown\", and then " +
                 "re-run your command.").format(xcode_version_flag),
            )
    return alias_to_versions.get(str(explicit_default_version.version)).xcode_version_properties

def _dotted_version_or_default(field, default):
    return _builtins.internal.apple_common.dotted_version(field) or default
