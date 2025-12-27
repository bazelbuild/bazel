// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.packages.util;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.lib.rules.python.PythonTestUtils.getPyLoad;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.io.IOException;
import java.util.stream.Stream;

/** Creates mock BUILD files required for the objc rules. */
public final class MockObjcSupport {

  private static final ImmutableList<String> OSX_ARCHS =
      ImmutableList.of(
          "x64_windows",
          "ios_x86_64",
          "ios_i386",
          "ios_armv7",
          "ios_arm64",
          "ios_arm64e",
          "darwin_x86_64",
          "darwin_arm64",
          "watchos_i386",
          "watchos_x86_64",
          "watchos_armv7k",
          "watchos_arm64_32",
          "tvos_x86_64",
          "tvos_arm64");

  public static final String DEFAULT_OSX_CROSSTOOL_DIR = "tools/osx/crosstool";
  private static final String MOCK_OSX_TOOLCHAIN_CONFIG_PATH =
      "com/google/devtools/build/lib/packages/util/mock/osx_cc_toolchain_config.bzl";

  /** The label of the {@code xcode_config} target used in test to enumerate available xcodes. */
  public static final String XCODE_VERSION_CONFIG =
      TestConstants.TOOLS_REPOSITORY + "//tools/objc:host_xcodes";

  /** The build label for the mock OSX crosstool configuration. */
  public static final String DEFAULT_OSX_CROSSTOOL =
      "//" + DEFAULT_OSX_CROSSTOOL_DIR + ":crosstool";

  public static final String DEFAULT_XCODE_VERSION = "7.3.1";
  public static final String DEFAULT_IOS_SDK_VERSION = "8.4";

  public static final String APPLE_SIMULATOR_PLATFORM_PACKAGE =
      TestConstants.APPLE_PLATFORM_PACKAGE_ROOT
          + (TestConstants.PRODUCT_NAME.equals("bazel") ? "" : "/simulator");

  public static final String DARWIN_X86_64 =
      TestConstants.APPLE_PLATFORM_PACKAGE_ROOT + ":darwin_x86_64";
  public static final String DARWIN_ARM64 =
      TestConstants.APPLE_PLATFORM_PACKAGE_ROOT + ":darwin_arm64";
  public static final String IOS_X86_64 = APPLE_SIMULATOR_PLATFORM_PACKAGE + ":ios_x86_64";
  public static final String IOS_ARM64 = TestConstants.APPLE_PLATFORM_PACKAGE_ROOT + ":ios_arm64";

  public static final String IOS_ARM64E = TestConstants.APPLE_PLATFORM_PACKAGE_ROOT + ":ios_arm64e";
  public static final String IOS_ARMV7 =
      TestConstants.APPLE_PLATFORM_PACKAGE_ROOT + ":ios_armv7"; // legacy for testing
  public static final String IOS_I386 =
      APPLE_SIMULATOR_PLATFORM_PACKAGE + ":ios_i386"; // legacy for testing
  public static final String WATCHOS_ARMV7K =
      TestConstants.APPLE_PLATFORM_PACKAGE_ROOT + ":watchos_armv7k";

  public static final String WATCHOS_ARM64_32 =
      TestConstants.APPLE_PLATFORM_PACKAGE_ROOT + ":watchos_arm64_32";

  public static ImmutableList<String> requiredObjcPlatformFlags(String... args) {
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    return builder
        .addAll(requiredObjcPlatformFlagsNoXcodeConfig(args))
        .add("--xcode_version_config=" + MockObjcSupport.XCODE_VERSION_CONFIG)
        .build();
  }

  /** Returns the set of flags required to build objc libraries using the mock OSX crosstool. */
  public static ImmutableList<String> requiredObjcCrosstoolFlags(String... args) {
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    return builder
        .addAll(requiredObjcCrosstoolFlagsNoXcodeConfig(args))
        .add("--xcode_version_config=" + MockObjcSupport.XCODE_VERSION_CONFIG)
        .build();
  }

  public static ImmutableList<String> requiredObjcPlatformFlagsNoXcodeConfig(String... args) {
    ImmutableList.Builder<String> argsBuilder = ImmutableList.builder();
    argsBuilder.addAll(Stream.of(args).collect(toImmutableList()));
    if (Stream.of(args).noneMatch(arg -> arg.startsWith("--platforms="))) {
      argsBuilder.add("--platforms=" + MockObjcSupport.DARWIN_X86_64);
    }

    return argsBuilder.build();
  }

  /**
   * Returns the set of flags required to build objc libraries using the mock OSX crosstool except
   * for --xcode_version_config.
   */
  public static ImmutableList<String> requiredObjcCrosstoolFlagsNoXcodeConfig(String... args) {

    ImmutableList.Builder<String> argsBuilder = ImmutableList.builder();
    argsBuilder.addAll(Stream.of(args).collect(toImmutableList()));
    if (Stream.of(args).noneMatch(arg -> arg.startsWith("--platforms="))) {
      argsBuilder.add("--platforms=" + MockObjcSupport.DARWIN_X86_64);
    }

    return argsBuilder.build();
  }

  public static void setupXcodeRules(MockToolsConfig config) throws IOException {
    config.create("build_bazel_apple_support/xcode/BUILD");
    config.create(
        "build_bazel_apple_support/xcode/xcode_version.bzl",
        """
        XcodeVersionRuleInfo = provider(fields = ["aliases", "label", "xcode_version_properties"])

        def _xcode_version_properties_info_init(
                *,
                xcode_version,
                default_ios_sdk_version = "8.4",
                default_macos_sdk_version = "10.11",
                default_tvos_sdk_version = "9.0",
                default_watchos_sdk_version = "2.0",
                default_visionos_sdk_version = "1.0"):
            return {
                "xcode_version": xcode_version,
                "default_ios_sdk_version": default_ios_sdk_version,
                "default_macos_sdk_version": default_macos_sdk_version,
                "default_tvos_sdk_version": default_tvos_sdk_version,
                "default_watchos_sdk_version": default_watchos_sdk_version,
                "default_visionos_sdk_version": default_visionos_sdk_version,
            }

        XcodeVersionPropertiesInfo, _new_xcode_version_properties_info = provider(
            fields = [
                "xcode_version",
                "default_ios_sdk_version",
                "default_macos_sdk_version",
                "default_tvos_sdk_version",
                "default_watchos_sdk_version",
                "default_visionos_sdk_version",
            ],
            init = _xcode_version_properties_info_init,
        )

        def _xcode_version_impl(ctx):
            xcode_version_properties = XcodeVersionPropertiesInfo(
                xcode_version = ctx.attr.version,
                default_ios_sdk_version = ctx.attr.default_ios_sdk_version,
                default_visionos_sdk_version = ctx.attr.default_visionos_sdk_version,
                default_watchos_sdk_version = ctx.attr.default_watchos_sdk_version,
                default_tvos_sdk_version = ctx.attr.default_tvos_sdk_version,
                default_macos_sdk_version = ctx.attr.default_macos_sdk_version,
            )
            return [
                xcode_version_properties,
                XcodeVersionRuleInfo(
                    label = ctx.label,
                    xcode_version_properties = xcode_version_properties,
                    aliases = ctx.attr.aliases,
                ),
                DefaultInfo(runfiles = ctx.runfiles()),
            ]

        xcode_version = rule(
            attrs = {
                "version": attr.string(mandatory = True),
                "aliases": attr.string_list(allow_empty = True, mandatory = False),
                "default_ios_sdk_version": attr.string(default = "8.4", mandatory = False),
                "default_visionos_sdk_version": attr.string(default = "1.0", mandatory = False),
                "default_watchos_sdk_version": attr.string(default = "2.0", mandatory = False),
                "default_tvos_sdk_version": attr.string(default = "9.0",  mandatory = False),
                "default_macos_sdk_version": attr.string(default = "10.11",  mandatory = False),
            },
            implementation = _xcode_version_impl,
        )
        """);
    config.create(
        "build_bazel_apple_support/xcode/available_xcodes.bzl",
        """
        load(":xcode_version.bzl", "XcodeVersionRuleInfo")
        AvailableXcodesInfo = provider(fields = ["available_versions", "default_version"])

        def _available_xcodes_impl(ctx):
            available_versions = [
                target[XcodeVersionRuleInfo]
                for target in ctx.attr.versions
            ]
            default_version = ctx.attr.default[XcodeVersionRuleInfo]

            return [
                AvailableXcodesInfo(
                    available_versions = available_versions,
                    default_version = default_version,
                ),
            ]

        available_xcodes = rule(
            attrs = {
                "default": attr.label(
                    doc = "The default Xcode version for this platform.",
                    mandatory = True,
                    providers = [[XcodeVersionRuleInfo]],
                    flags = ["NONCONFIGURABLE"],
                ),
                "versions": attr.label_list(
                    doc = "The Xcode versions that are available on this platform.",
                    providers = [[XcodeVersionRuleInfo]],
                    flags = ["NONCONFIGURABLE"],
                ),
            },
            implementation = _available_xcodes_impl,
            provides = [AvailableXcodesInfo],
        )
        """);
    config.create(
        "build_bazel_apple_support/xcode/xcode_config.bzl",
        """
        load(":available_xcodes.bzl", "AvailableXcodesInfo")
        load(":xcode_version.bzl", "XcodeVersionRuleInfo", "XcodeVersionPropertiesInfo")

        XcodeVersionInfo = apple_common.XcodeVersionConfig
        unavailable_xcode_message = "'bazel fetch --force --configure' (Bzlmod) or 'bazel sync --configure' (WORKSPACE)"

        def _xcode_config_impl(ctx):
            apple_fragment = ctx.fragments.apple
            cpp_fragment = ctx.fragments.cpp

            explicit_default_version = ctx.attr.default[XcodeVersionRuleInfo] if ctx.attr.default else None
            explicit_versions = [
                target[XcodeVersionRuleInfo]
                for target in ctx.attr.versions
            ] if ctx.attr.versions else []
            remote_versions = [
                target
                for target in ctx.attr.remote_versions[AvailableXcodesInfo].available_versions
            ] if ctx.attr.remote_versions else []
            local_versions = [
                target
                for target in ctx.attr.local_versions[AvailableXcodesInfo].available_versions
            ] if ctx.attr.local_versions else []

            local_default_version = ctx.attr.local_versions[AvailableXcodesInfo].default_version if ctx.attr.local_versions else None
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
                visionos_minimum_os = apple_common.dotted_version(cpp_fragment.minimum_os_version())
            else:
                visionos_minimum_os = visionos_sdk_version

            xcode_versions = XcodeVersionInfo(
                ios_sdk_version = str(ios_sdk_version),
                ios_minimum_os_version = str(ios_minimum_os),
                visionos_sdk_version = str(visionos_sdk_version),
                visionos_minimum_os_version = str(visionos_minimum_os),
                watchos_sdk_version = str(watchos_sdk_version),
                watchos_minimum_os_version = str(watchos_minimum_os),
                tvos_sdk_version = str(tvos_sdk_version),
                tvos_minimum_os_version = str(tvos_minimum_os),
                macos_sdk_version = str(macos_sdk_version),
                macos_minimum_os_version = str(macos_minimum_os),
                xcode_version = xcode_version_properties.xcode_version,
                availability = availability,
                xcode_version_flag = apple_fragment.xcode_version_flag,
                include_xcode_execution_info = apple_fragment.include_xcode_exec_requirements,
            )
            return [
                DefaultInfo(runfiles = ctx.runfiles()),
                xcode_versions,
                xcode_version_properties,
            ]

        xcode_config = rule(
            attrs = {
                "default": attr.label(
                    flags = ["NONCONFIGURABLE"],
                    providers = [[XcodeVersionRuleInfo]],
                ),
                "versions": attr.label_list(
                    flags = ["NONCONFIGURABLE"],
                    providers = [[XcodeVersionRuleInfo]],
                ),
                "remote_versions": attr.label(
                    flags = ["NONCONFIGURABLE"],
                    providers = [[AvailableXcodesInfo]],
                ),
                "local_versions": attr.label(
                    flags = ["NONCONFIGURABLE"],
                    providers = [[AvailableXcodesInfo]],
                ),
            },
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
                if alias in version.aliases or (version.xcode_version_properties.xcode_version == alias):
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
                version_string = version.xcode_version_properties.xcode_version
                if version_string not in version.aliases:  # only add the version if it's not also an alias
                    if version_string in version_map:
                        fail(_duplicate_alias_error(version_string, versions))
                    else:
                        version_map[version_string] = version
            return version_map

        def _resolve_xcode_from_local_and_remote(
                local_versions,
                remote_versions,
                xcode_version_flag,
                prefer_mutual_xcode,
                local_default_version):
            local_alias_to_version_map = _aliases_to_xcode_version(local_versions)
            remote_alias_to_version_map = _aliases_to_xcode_version(remote_versions)

            mutually_available_versions = {}
            for version in local_versions:
                version_string = version.xcode_version_properties.xcode_version
                if version_string in remote_alias_to_version_map:
                    mutually_available_versions[version_string] = remote_alias_to_version_map[version_string]

            # We'd log an event here if we could!!
            if xcode_version_flag:
                remote_version_from_flag = remote_alias_to_version_map.get(xcode_version_flag)
                local_version_from_flag = local_alias_to_version_map.get(xcode_version_flag)
                availability = "BOTH"

                if remote_version_from_flag and local_version_from_flag:
                    local_version_from_remote_versions = remote_alias_to_version_map.get(local_version_from_flag.xcode_version_properties.xcode_version)
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
                                local_version_from_flag.xcode_version_properties.xcode_version,
                                remote_version_from_flag.xcode_version_properties.xcode_version,
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
                            ", ".join([version.xcode_version_properties.xcode_version for version in local_versions]),
                            ", ".join([version.xcode_version_properties.xcode_version for version in remote_versions]),
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
                    " performance.").format(local_default_version.xcode_version_properties.xcode_version, ", ".join([version.xcode_version_properties.xcode_version for version in remote_versions])),
                )
                local_version = local_default_version
                availability = "LOCAL"
            elif (local_default_version.xcode_version_properties.xcode_version in remote_alias_to_version_map):
                #  If the local default version is also available remotely, use it.
                availability = "BOTH"
                local_version = remote_alias_to_version_map.get(local_default_version.xcode_version_properties.xcode_version)
            else:
                # If an alias of the local default version is available remotely, use it.
                for version_number in local_default_version.aliases:
                    if version_number in remote_alias_to_version_map:
                        availability = "BOTH"
                        local_version = remote_alias_to_version_map.get(version_number)
                        break

            if local_version:
                return local_version.xcode_version_properties, availability

            # The local default is not available remotely.
            if prefer_mutual_xcode:
                # If we prefer a mutually available version, the newest one.
                newest_version = "0.0"
                default_version = None
                for _, version in mutually_available_versions.items():
                    if _compare_version_strings(version.xcode_version_properties.xcode_version, newest_version) > 0:
                        default_version = version
                        newest_version = default_version.xcode_version_properties.xcode_version

                return default_version.xcode_version_properties, "BOTH"
            else:
                # Use the local default
                return local_default_version.xcode_version_properties, "LOCAL"

        def _compare_version_strings(first, second):
            return apple_common.dotted_version(first).compare_to(apple_common.dotted_version(second))

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
                return XcodeVersionPropertiesInfo(xcode_version = None)

            if not explicit_default_version:
                fail("if any versions are specified, a default version must be specified")

            alias_to_versions = _aliases_to_xcode_version(explicit_versions)
            if xcode_version_flag:
                flag_version = alias_to_versions.get(str(xcode_version_flag))
                if flag_version:
                    return flag_version.xcode_version_properties
                elif xcode_version_flag.startswith("/"):
                    return XcodeVersionPropertiesInfo(xcode_version = xcode_version_flag)
                else:
                    fail(
                        ("--xcode_version={0} specified, but '{0}' is not an available Xcode version. " +
                        "If you believe you have '{0}' installed, try running 'bazel shutdown', and then " +
                        "re-run your command.").format(xcode_version_flag),
                    )

            return alias_to_versions.get(explicit_default_version.xcode_version_properties.xcode_version).xcode_version_properties

        def _dotted_version_or_default(field, default):
            return apple_common.dotted_version(field) or default
        """);
    config.create(
        "build_bazel_apple_support/xcode/xcode_config_alias.bzl",
        """
        load(":xcode_version.bzl", "XcodeVersionPropertiesInfo")

        def _xcode_config_alias_impl(ctx):
            xcode_config = ctx.attr._xcode_config
            return [
                xcode_config[XcodeVersionPropertiesInfo],
                xcode_config[apple_common.XcodeVersionConfig],
            ]

        xcode_config_alias = rule(
            attrs = {
                "_xcode_config": attr.label(
                    default = configuration_field(
                        fragment = "apple",
                        name = "xcode_config_label",
                    ),
                ),
            },
            fragments = ["apple"],
            implementation = _xcode_config_alias_impl,
        )
        """);
  }

  /**
   * Sets up the support for building ObjC. Any partial toolchain line will be merged into every
   * toolchain stanza in the crosstool loaded from file.
   */
  public static void setup(MockToolsConfig config) throws IOException {

    // Create default, simple Apple toolchains based on the default Apple Crosstool.
    config.create(
        "tools/build_defs/apple/toolchains/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "toolchain(",
        "  name = 'darwin_x86_64_any',",
        "  toolchain = '//"
            + MockObjcSupport.DEFAULT_OSX_CROSSTOOL_DIR
            + ":cc-compiler-darwin_x86_64',",
        "  toolchain_type = '" + TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type',",
        "  target_compatible_with = [",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:osx',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_64',",
        "  ],",
        ")",
        "toolchain(",
        "  name = 'ios_arm64_any',",
        "  toolchain = '//"
            + MockObjcSupport.DEFAULT_OSX_CROSSTOOL_DIR
            + ":cc-compiler-ios_arm64',",
        "  toolchain_type = '" + TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type',",
        "  target_compatible_with = [",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:ios',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:arm64',",
        "  ],",
        ")");

    // Create default Apple target platforms.
    // Any device, simulator or maccatalyst platforms created by Apple tests should consider
    // building on one of these targets as parents, to ensure that the proper constraints are set.
    config.create(
        TestConstants.APPLE_PLATFORM_PATH + "/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "licenses(['notice'])",
        "platform(",
        "  name = 'darwin_x86_64',",
        "  constraint_values = [",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:osx',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_64',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "env:device',",
        "  ],",
        ")",
        "platform(",
        "  name = 'darwin_arm64',",
        "  constraint_values = [",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:osx',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:arm64',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "env:device',",
        "  ],",
        ")",
        "platform(",
        "  name = 'ios_arm64',",
        "  constraint_values = [",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:ios',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:arm64',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "env:device',",
        "  ],",
        ")",
        "platform(",
        "  name = 'ios_arm64e',",
        "  constraint_values = [",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:ios',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:arm64e',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "env:device',",
        "  ],",
        ")",
        "platform(", // legacy platform only used to support tests
        "  name = 'ios_armv7',",
        "  constraint_values = [",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:ios',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:armv7',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "env:device',",
        "  ],",
        ")",
        "platform(",
        "  name = 'watchos_armv7k',",
        "  constraint_values = [",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:watchos',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:armv7k',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "env:device',",
        "  ],",
        ")",
        "platform(",
        "  name = 'watchos_arm64_32',",
        "  constraint_values = [",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:watchos',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:arm64_32',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "env:device',",
        "  ],",
        ")");

    String[] simulatorPlatforms = {
      "platform(",
      "  name = 'ios_x86_64',",
      "  constraint_values = [",
      "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:ios',",
      "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_64',",
      "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "env:simulator',",
      "  ],",
      ")",
      "platform(",
      "  name = 'ios_i386',", // legacy platform only used to support tests
      "  constraint_values = [",
      "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:ios',",
      "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_32',",
      "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "env:simulator',",
      "  ],",
      ")",
      "platform(",
      "  name = 'watchos_x86_64',",
      "  constraint_values = [",
      "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:watchos',",
      "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_64',",
      "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "env:simulator',",
      "  ],",
      ")"
    };

    if (TestConstants.PRODUCT_NAME.equals("bazel")) {
      config.append(TestConstants.APPLE_PLATFORM_PATH + "/BUILD", simulatorPlatforms);
    } else {
      config.create(TestConstants.APPLE_PLATFORM_PATH + "/simulator/BUILD", simulatorPlatforms);
    }

    for (String tool : ImmutableSet.of("gcov", "testrunner", "mcov", "libtool")) {
      config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/objc/" + tool);
    }
    setupXcodeRules(config);
    config.create(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/objc/BUILD",
        getPyLoad("py_binary"),
        "load('@build_bazel_apple_support//xcode:xcode_version.bzl', 'xcode_version')",
        "load('@build_bazel_apple_support//xcode:xcode_config.bzl', 'xcode_config')",
        "package(default_visibility=['//visibility:public'])",
        "exports_files(glob(['**']))",
        "filegroup(name = 'default_provisioning_profile', srcs = ['foo.mobileprovision'])",
        "filegroup(name = 'xctest_infoplist', srcs = ['xctest.plist'])",
        "xcode_config(name = 'host_xcodes',",
        "  default = ':version7_3_1',",
        "  versions = [':version7_3_1', ':version5_0', ':version7_3', ':version5_8', ':version5'])",
        "xcode_version(",
        "  name = 'version7_3_1',",
        "  version = '" + DEFAULT_XCODE_VERSION + "',",
        "  default_ios_sdk_version = \"" + DEFAULT_IOS_SDK_VERSION + "\",",
        ")",
        "xcode_version(",
        "  name = 'version7_3',",
        "  version = '7.3',",
        ")",
        "xcode_version(",
        "  name = 'version5_0',",
        "  version = '5.0',",
        ")",
        "xcode_version(",
        "  name = 'version5_8',",
        "  version = '5.8',",
        ")",
        "xcode_version(",
        "  name = 'version5',",
        "  version = '5',",
        ")");
    // If the bazel tools repository is not in the workspace, also create a workspace tools/objc
    // package with a few lingering dependencies.
    // TODO(b/64537078): Move these dependencies underneath the tools workspace.
    if (TestConstants.TOOLS_REPOSITORY_SCRATCH.length() > 0) {
      config.create(
          "tools/objc/BUILD",
          """
          package(default_visibility = ["//visibility:public"])

          exports_files(glob(["**"]))

          filegroup(
              name = "default_provisioning_profile",
              srcs = ["foo.mobileprovision"],
          )

          filegroup(
              name = "xctest_infoplist",
              srcs = ["xctest.plist"],
          )
          """);
    }
    config.create(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/objc/foo.mobileprovision", "No such luck");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/objc/xctest.plist");
    setupCcToolchainConfig(config);
  }

  public static void setupCcToolchainConfig(
      MockToolsConfig config, CcToolchainConfig.Builder ccToolchainConfig) throws IOException {
    if (config.isRealFileSystem()) {
      config.linkTools(DEFAULT_OSX_CROSSTOOL_DIR);
    } else {
      CcToolchainConfig toolchainConfig = ccToolchainConfig.build();
      ImmutableList.Builder<CcToolchainConfig> toolchainConfigBuilder = ImmutableList.builder();
      toolchainConfigBuilder.add(toolchainConfig);
      if (!toolchainConfig.getTargetCpu().equals("darwin_x86_64")) {
        toolchainConfigBuilder.add(darwinX86_64().build());
      }

      new Crosstool(
              config,
              DEFAULT_OSX_CROSSTOOL_DIR,
              Label.parseCanonicalUnchecked("@bazel_tools//tools/osx"))
          .setCcToolchainFile(readCcToolchainConfigFile())
          .setSupportedArchs(OSX_ARCHS)
          .setToolchainConfigs(toolchainConfigBuilder.build())
          .setSupportsHeaderParsing(true)
          .writeOSX();
    }
  }

  public static void setupCcToolchainConfig(MockToolsConfig config) throws IOException {
    if (config.isRealFileSystem()) {
      config.linkTools(DEFAULT_OSX_CROSSTOOL_DIR);
    } else {
      new Crosstool(
              config,
              DEFAULT_OSX_CROSSTOOL_DIR,
              Label.parseCanonicalUnchecked("@bazel_tools//tools/osx"))
          .setCcToolchainFile(readCcToolchainConfigFile())
          .setSupportedArchs(OSX_ARCHS)
          .setToolchainConfigs(getDefaultCcToolchainConfigs())
          .setSupportsHeaderParsing(true)
          .writeOSX();
    }
  }

  private static ImmutableList<CcToolchainConfig> getDefaultCcToolchainConfigs() {
    return ImmutableList.of(
        darwinX86_64().build(),
        darwin_arm64().build(),
        x64_windows().build(),
        ios_arm64().build(),
        ios_arm64e().build(),
        ios_armv7().build(),
        ios_i386().build(),
        iosX86_64().build(),
        tvos_arm64().build(),
        tvosX86_64().build(),
        watchos_armv7k().build(),
        watchos_arm64_32().build(),
        watchos_i386().build(),
        watchosX86_64().build());
  }

  public static CcToolchainConfig.Builder darwinX86_64() {
    return CcToolchainConfig.builder()
        .withCpu("darwin_x86_64")
        .withCompiler("compiler")
        .withToolchainIdentifier("darwin_x86_64")
        .withHostSystemName("x86_64-apple-macosx")
        .withTargetSystemName("x86_64-apple-macosx")
        .withTargetLibc("macosx")
        .withAbiVersion("darwin_x86_64")
        .withAbiLibcVersion("darwin_x86_64")
        .withCcTargetOs("apple")
        .withSysroot("")
        .withCxxBuiltinIncludeDirectories(
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/Applications/Xcode_7.2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/Applications/Xcode_8.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/Applications/Xcode_8.1.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/usr/include")
        .withToolchainTargetConstraints(
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_64",
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:osx");
  }

  public static CcToolchainConfig.Builder darwin_arm64() {
    return CcToolchainConfig.builder()
        .withCpu("darwin_arm64")
        .withCompiler("compiler")
        .withToolchainIdentifier("darwin_arm64")
        .withHostSystemName("x86_64-apple-macosx")
        .withTargetSystemName("arm64-apple-macosx")
        .withTargetLibc("macosx")
        .withAbiVersion("local")
        .withAbiLibcVersion("local")
        .withCcTargetOs("apple")
        .withSysroot("")
        .withCxxBuiltinIncludeDirectories(
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/Applications/Xcode_7.2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/Applications/Xcode_8.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/Applications/Xcode_8.1.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs",
            "/usr/include")
        .withToolchainTargetConstraints(
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:arm64",
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:osx")
        .withToolchainExecConstraints(
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:arm64",
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:osx");
  }

  public static CcToolchainConfig.Builder x64_windows() {
    return CcToolchainConfig.builder()
        .withCpu("x64_windows")
        .withCompiler("compiler")
        .withToolchainIdentifier("x64_windows")
        .withHostSystemName("x86_64-apple-macosx")
        .withTargetSystemName("x86_64-apple-macosx")
        .withTargetLibc("macosx")
        .withAbiVersion("darwin_x86_64")
        .withAbiLibcVersion("darwin_x86_64")
        .withCcTargetOs("apple")
        .withSysroot("")
        .withCxxBuiltinIncludeDirectories(
            "/Applications/<xcode_version>/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/<xcode_version>/Contents/Developer/Platforms/<platform>.platform/Developer/SDKs",
            "/usr/include");
  }

  public static CcToolchainConfig.Builder ios_arm64() {
    return CcToolchainConfig.builder()
        .withCpu("ios_arm64")
        .withCompiler("compiler")
        .withToolchainIdentifier("ios_arm64")
        .withHostSystemName("x86_64-apple-macosx")
        .withTargetSystemName("arm64-apple-ios")
        .withTargetLibc("ios")
        .withAbiVersion("local")
        .withAbiLibcVersion("local")
        .withCcTargetOs("apple")
        .withSysroot("")
        .withCxxBuiltinIncludeDirectories(
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode-beta.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_7.2.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.1.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/usr/include")
        .withToolchainTargetConstraints(
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:arm64",
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:ios");
  }

  public static CcToolchainConfig.Builder ios_arm64e() {
    return CcToolchainConfig.builder()
        .withCpu("ios_arm64e")
        .withCompiler("compiler")
        .withToolchainIdentifier("ios_arm64e")
        .withHostSystemName("x86_64e-apple-macosx")
        .withTargetSystemName("arm64e-apple-ios")
        .withTargetLibc("ios")
        .withAbiVersion("local")
        .withAbiLibcVersion("local")
        .withCcTargetOs("apple")
        .withSysroot("")
        .withCxxBuiltinIncludeDirectories(
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode-beta.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_7.2.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.1.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/usr/include")
        .withToolchainTargetConstraints(
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:arm64e",
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:ios");
  }

  public static CcToolchainConfig.Builder ios_armv7() {
    return CcToolchainConfig.builder()
        .withCpu("ios_armv7")
        .withCompiler("compiler")
        .withToolchainIdentifier("ios_armv7")
        .withHostSystemName("x86_64-apple-macosx")
        .withTargetSystemName("armv7-apple-ios")
        .withTargetLibc("ios")
        .withAbiVersion("local")
        .withAbiLibcVersion("local")
        .withCcTargetOs("apple")
        .withSysroot("")
        .withCxxBuiltinIncludeDirectories(
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode-beta.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_7.2.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.1.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs",
            "/usr/include")
        .withToolchainTargetConstraints(
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:armv7",
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:ios");
  }

  public static CcToolchainConfig.Builder ios_i386() {
    return CcToolchainConfig.builder()
        .withCpu("ios_i386")
        .withCompiler("compiler")
        .withToolchainIdentifier("ios_i386")
        .withHostSystemName("x86_64-apple-macosx")
        .withTargetSystemName("i386-apple-ios")
        .withTargetLibc("ios")
        .withAbiVersion("local")
        .withAbiLibcVersion("local")
        .withCcTargetOs("apple")
        .withSysroot("")
        .withCxxBuiltinIncludeDirectories(
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/Applications/Xcode-beta.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_7.2.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.1.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/usr/include")
        .withToolchainTargetConstraints(
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_32",
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:ios");
  }

  public static CcToolchainConfig.Builder iosX86_64() {
    return CcToolchainConfig.builder()
        .withCpu("ios_x86_64")
        .withCompiler("compiler")
        .withToolchainIdentifier("ios_x86_64")
        .withHostSystemName("x86_64-apple-ios")
        .withTargetSystemName("x86_64-apple-ios")
        .withTargetLibc("ios")
        .withAbiVersion("local")
        .withAbiLibcVersion("local")
        .withCcTargetOs("apple")
        .withSysroot("")
        .withCxxBuiltinIncludeDirectories(
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/Applications/Xcode-beta.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_7.2.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.1.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs",
            "/usr/include")
        .withToolchainTargetConstraints(
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_64",
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:ios");
  }

  public static CcToolchainConfig.Builder tvos_arm64() {
    return CcToolchainConfig.builder()
        .withCpu("tvos_arm64")
        .withCompiler("compiler")
        .withToolchainIdentifier("tvos_arm64")
        .withHostSystemName("x86_64-apple-ios")
        .withTargetSystemName("arm64-apple-tvos")
        .withTargetLibc("tvos")
        .withAbiVersion("local")
        .withAbiLibcVersion("local")
        .withCcTargetOs("apple")
        .withSysroot("")
        .withCxxBuiltinIncludeDirectories(
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode.app/Contents/Developer/Platforms/AppleTVOS.platform/Developer/SDKs",
            "/Applications/Xcode-beta.app/Contents/Developer/Platforms/AppleTVOS.platform/Developer/SDKs",
            "/Applications/Xcode_7.2.app/Contents/Developer/Platforms/AppleTVOS.platform/Developer/SDKs",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Platforms/AppleTVOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.app/Contents/Developer/Platforms/AppleTVOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.1.app/Contents/Developer/Platforms/AppleTVOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.app/Contents/Developer/Platforms/AppleTVOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Platforms/AppleTVOS.platform/Developer/SDKs",
            "/usr/include")
        .withToolchainTargetConstraints(
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:arm64",
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:tvos");
  }

  public static CcToolchainConfig.Builder tvosX86_64() {
    return CcToolchainConfig.builder()
        .withCpu("tvos_x86_64")
        .withCompiler("compiler")
        .withToolchainIdentifier("tvos_x86_64")
        .withHostSystemName("x86_64-apple-ios")
        .withTargetSystemName("x86_64-apple-tvos")
        .withTargetLibc("tvos")
        .withAbiVersion("local")
        .withAbiLibcVersion("local")
        .withCcTargetOs("apple")
        .withSysroot("")
        .withCxxBuiltinIncludeDirectories(
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode.app/Contents/Developer/Platforms/AppleTVSimulator.platform/Developer/SDKs",
            "/Applications/Xcode-beta.app/Contents/Developer/Platforms/AppleTVSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_7.2.app/Contents/Developer/Platforms/AppleTVSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Platforms/AppleTVSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.app/Contents/Developer/Platforms/AppleTVSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.1.app/Contents/Developer/Platforms/AppleTVSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.app/Contents/Developer/Platforms/AppleTVSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Platforms/AppleTVSimulator.platform/Developer/SDKs",
            "/usr/include")
        .withToolchainTargetConstraints(
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_64",
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:tvos");
  }

  public static CcToolchainConfig.Builder watchos_armv7k() {
    return CcToolchainConfig.builder()
        .withCpu("watchos_armv7k")
        .withCompiler("compiler")
        .withToolchainIdentifier("watchos_armv7k")
        .withHostSystemName("x86_64-apple-ios")
        .withTargetSystemName("armv7k-apple-watchos")
        .withTargetLibc("watchos")
        .withAbiVersion("local")
        .withAbiLibcVersion("local")
        .withCcTargetOs("apple")
        .withSysroot("")
        .withCxxBuiltinIncludeDirectories(
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/Applications/Xcode-beta.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/Applications/Xcode_7.2.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.1.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/usr/include")
        .withToolchainTargetConstraints(
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:armv7k",
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:watchos");
  }

  @SuppressWarnings("MemberName") // Following style of other mock toolchain config methods.
  public static CcToolchainConfig.Builder watchos_arm64_32() {
    return CcToolchainConfig.builder()
        .withCpu("watchos_arm64_32")
        .withCompiler("compiler")
        .withToolchainIdentifier("watchos_arm64_32")
        .withHostSystemName("x86_64-apple-ios")
        .withTargetSystemName("arm64_32-apple-watchos")
        .withTargetLibc("watchos")
        .withAbiVersion("local")
        .withAbiLibcVersion("local")
        .withCcTargetOs("apple")
        .withSysroot("")
        .withCxxBuiltinIncludeDirectories(
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/Applications/Xcode-beta.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/Applications/Xcode_7.2.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.1.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Platforms/WatchOS.platform/Developer/SDKs",
            "/usr/include")
        .withToolchainTargetConstraints(
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:arm64_32",
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:watchos");
  }

  public static CcToolchainConfig.Builder watchos_i386() {
    return CcToolchainConfig.builder()
        .withCpu("watchos_i386")
        .withCompiler("compiler")
        .withToolchainIdentifier("watchos_i386")
        .withHostSystemName("x86_64-apple-ios")
        .withTargetSystemName("i386-apple-watchos")
        .withTargetLibc("watchos")
        .withAbiVersion("local")
        .withAbiLibcVersion("local")
        .withCcTargetOs("apple")
        .withSysroot("")
        .withCxxBuiltinIncludeDirectories(
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/Applications/Xcode-beta.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_7.2.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.1.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/usr/include")
        .withToolchainTargetConstraints(
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_32",
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:watchos");
  }

  @SuppressWarnings("MemberName") // Following style of other mock toolchain config methods.
  public static CcToolchainConfig.Builder watchosX86_64() {
    return CcToolchainConfig.builder()
        .withCpu("watchos_x86_64")
        .withCompiler("compiler")
        .withToolchainIdentifier("watchos_x86_64")
        .withHostSystemName("x86_64-apple-ios")
        .withTargetSystemName("x86_64-apple-watchos")
        .withTargetLibc("watchos")
        .withAbiVersion("local")
        .withAbiLibcVersion("local")
        .withCcTargetOs("apple")
        .withSysroot("")
        .withCxxBuiltinIncludeDirectories(
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/",
            "/Applications/Xcode.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/Applications/Xcode-beta.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_7.2.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_7.3.1.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.1.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/Applications/Xcode_8.2.1.app/Contents/Developer/Platforms/WatchSimulator.platform/Developer/SDKs",
            "/usr/include")
        .withToolchainTargetConstraints(
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_64",
            TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:watchos");
  }

  /** Test setup for the Apple SDK targets that are used in tests. */
  public static void setupAppleSdks(MockToolsConfig config) throws IOException {
    config.create(
        "third_party/apple_sdks/BUILD",
        "package(default_visibility=['//visibility:public'])\n"
            + "licenses([\"notice\"])\n"
            + "filegroup(name = \"apple_sdk_compile\")");
  }

  public static String readCcToolchainConfigFile() throws IOException {
    return ResourceLoader.readFromResources(MOCK_OSX_TOOLCHAIN_CONFIG_PATH);
  }
}
