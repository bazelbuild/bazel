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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.io.IOException;

/**
 * Creates mock BUILD files required for the objc rules.
 */
public final class MockObjcSupport {

  private static final ImmutableList<String> OSX_ARCHS =
      ImmutableList.of(
          "x64_windows",
          "ios_x86_64",
          "ios_i386",
          "ios_armv7",
          "ios_arm64",
          "darwin_x86_64",
          "watchos_i386",
          "watchos_armv7k",
          "tvos_x86_64",
          "tvos_arm64");

  private static final ImmutableList<String> DEFAULT_OSX_CROSSTOOL_DEPS_DIRS =
      ImmutableList.of("third_party/bazel/tools/osx/crosstool");
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

  /** Returns the set of flags required to build objc libraries using the mock OSX crosstool. */
  public static ImmutableList<String> requiredObjcCrosstoolFlags() {
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    return builder
        .addAll(requiredObjcCrosstoolFlagsNoXcodeConfig())
        .add("--xcode_version_config=" + MockObjcSupport.XCODE_VERSION_CONFIG)
        .build();
  }

  /**
   * Returns the set of flags required to build objc libraries using the mock OSX crosstool except
   * for --xcode_version_config.
   */
  public static ImmutableList<String> requiredObjcCrosstoolFlagsNoXcodeConfig() {

    ImmutableList.Builder<String> argsBuilder = ImmutableList.builder();

    // TODO(b/68751876): Set --apple_crosstool_top and --crosstool_top using the
    // AppleCrosstoolTransition
    argsBuilder
        .add("--apple_crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL)
        .add("--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL);

    // TODO(b/32411441): This flag will be flipped off by default imminently, at which point
    // this can be removed. The flag itself is for safe rollout of a backwards incompatible change.
    argsBuilder.add("--noexperimental_objc_provider_from_linked");
    return argsBuilder.build();
  }

  /**
   * Sets up the support for building ObjC. Any partial toolchain line will be merged into every
   * toolchain stanza in the crosstool loaded from file.
   */
  public static void setup(MockToolsConfig config) throws IOException {
    for (String tool :
        ImmutableSet.of(
            "objc_dummy.mm",
            "gcov",
            "realpath",
            "testrunner",
            "xcrunwrapper.sh",
            "mcov",
            "libtool")) {
      config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/objc/" + tool);
    }
    config.create(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/objc/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "exports_files(glob(['**']))",
        "filegroup(name = 'default_provisioning_profile', srcs = ['foo.mobileprovision'])",
        "filegroup(name = 'compile_protos', srcs = ['compile_protos.py'])",
        "filegroup(name = 'protobuf_compiler_wrapper', srcs = ['protobuf_compiler_wrapper.sh'])",
        "filegroup(name = 'protobuf_compiler', srcs = ['protobuf_compiler_helper.py'])",
        "filegroup(",
        "  name = 'protobuf_compiler_support',",
        "  srcs = ['proto_support', 'protobuf_compiler_helper.py'],",
        ")",
        "sh_binary(name = 'xcrunwrapper', srcs = ['xcrunwrapper.sh'])",
        "apple_binary(name = 'xctest_appbin', platform_type = 'ios', deps = [':dummy_lib'])",
        "filegroup(name = 'xctest_infoplist', srcs = ['xctest.plist'])",
        "filegroup(name = 'j2objc_dead_code_pruner', srcs = ['j2objc_dead_code_pruner.py'])",
        "filegroup(",
        "  name = 'protobuf_well_known_types',",
        String.format(
            "  srcs = ['%s//objcproto:well_known_type.proto'],", TestConstants.TOOLS_REPOSITORY),
        ")",
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
        ")",
        "objc_library(name = 'dummy_lib', srcs = ['objc_dummy.mm'])",
        "alias(name = 'protobuf_lib', actual = '//objcproto:protobuf_lib')");
    // If the bazel tools repository is not in the workspace, also create a workspace tools/objc
    // package with a few lingering dependencies.
    // TODO(b/64537078): Move these dependencies underneath the tools workspace.
    if (TestConstants.TOOLS_REPOSITORY_SCRATCH.length() > 0) {
      config.create(
          "tools/objc/BUILD",
          "package(default_visibility=['//visibility:public'])",
          "exports_files(glob(['**']))",
          "apple_binary(name = 'xctest_appbin', platform_type = 'ios', deps = [':dummy_lib'])",
          "filegroup(name = 'default_provisioning_profile', srcs = ['foo.mobileprovision'])",
          "filegroup(name = 'xctest_infoplist', srcs = ['xctest.plist'])");
    }
    config.create(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/objc/foo.mobileprovision", "No such luck");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/objc/compile_protos.py");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/objc/xctest.plist");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/objc/proto_support");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/objc/j2objc_dead_code_pruner.py");
    setupCcToolchainConfig(config);
    setupObjcProto(config);
  }

  /** Sets up the support for building protocol buffers for ObjC. */
  private static void setupObjcProto(MockToolsConfig config) throws IOException {
    config.create(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "objcproto/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "objc_library(",
        "  name = 'protobuf_lib',",
        "  srcs = ['empty.m'],",
        "  hdrs = ['include/header.h'],",
        "  includes = ['include'],",
        ")",
        "exports_files(['well_known_type.proto'])",
        "proto_library(",
        "  name = 'well_known_type_proto',",
        "  srcs = ['well_known_type.proto'],",
        ")");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "objcproto/empty.m");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "objcproto/empty.cc");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "objcproto/well_known_type.proto");
  }

  public static void setupCcToolchainConfig(
      MockToolsConfig config, CcToolchainConfig.Builder ccToolchainConfig) throws IOException {
    if (config.isRealFileSystem()) {
      for (String depDir : DEFAULT_OSX_CROSSTOOL_DEPS_DIRS) {
        config.linkTools(depDir);
      }
      config.linkTools(DEFAULT_OSX_CROSSTOOL_DIR);
    } else {
      CcToolchainConfig toolchainConfig = ccToolchainConfig.build();
      ImmutableList.Builder<CcToolchainConfig> toolchainConfigBuilder = ImmutableList.builder();
      toolchainConfigBuilder.add(toolchainConfig);
      if (!toolchainConfig.getTargetCpu().equals("darwin_x86_64")) {
        toolchainConfigBuilder.add(darwinX86_64().build());
      }

      new Crosstool(config, DEFAULT_OSX_CROSSTOOL_DIR)
          .setCcToolchainFile(readCcToolchainConfigFile())
          .setSupportedArchs(OSX_ARCHS)
          .setToolchainConfigs(toolchainConfigBuilder.build())
          .setSupportsHeaderParsing(false)
          .writeOSX();
    }
  }

  public static void setupCcToolchainConfig(MockToolsConfig config) throws IOException {
    if (config.isRealFileSystem()) {
      for (String depDir : DEFAULT_OSX_CROSSTOOL_DEPS_DIRS) {
        config.linkTools(depDir);
      }
      config.linkTools(DEFAULT_OSX_CROSSTOOL_DIR);
    } else {
      new Crosstool(config, DEFAULT_OSX_CROSSTOOL_DIR)
          .setCcToolchainFile(readCcToolchainConfigFile())
          .setSupportedArchs(OSX_ARCHS)
          .setToolchainConfigs(getDefaultCcToolchainConfigs())
          .setSupportsHeaderParsing(false)
          .writeOSX();
    }
  }

  private static ImmutableList<CcToolchainConfig> getDefaultCcToolchainConfigs() {
    return ImmutableList.of(
        darwinX86_64().build(),
        x64_windows().build(),
        ios_arm64().build(),
        ios_armv7().build(),
        ios_i386().build(),
        iosX86_64().build(),
        tvos_arm64().build(),
        tvosX86_64().build(),
        watchos_armv7k().build(),
        watchos_i386().build());
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
            "/usr/include");
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
            "/usr/include");
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
            "/usr/include");
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
            "/usr/include");
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
            "/usr/include");
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
            "/usr/include");
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
            "/usr/include");
  }

  public static CcToolchainConfig.Builder watchos_armv7k() {
    return CcToolchainConfig.builder()
        .withCpu("watchos_armv7k")
        .withCompiler("compiler")
        .withToolchainIdentifier("watchos_armv7k")
        .withHostSystemName("x86_64-apple-ios")
        .withTargetSystemName("armv7-apple-watchos")
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
            "/usr/include");
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
            "/usr/include");
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

  /** Creates a mock objc_proto_library rule in the current main workspace. */
  public static void setupObjcProtoLibrary(Scratch scratch) throws Exception {
    // Append file instead of creating one in case it already exists.
    String toolsRepo = TestConstants.TOOLS_REPOSITORY;
    scratch.file("objc_proto_library/BUILD", "");
    scratch.file(
        "objc_proto_library/objc_proto_library.bzl",
        "def _impl(ctx):",
        "  return [apple_common.new_objc_provider()]",
        "",
        "objc_proto_library = rule(",
        "  _impl,",
        "  attrs = {",
        "    'deps': attr.label_list(),",
        "    'portable_proto_filters': attr.label_list(allow_files=True),",
        "    '_lib_protobuf': attr.label(default = '" + toolsRepo + "//objcproto:protobuf_lib'),",
        "  }",
        ")");
  }
}
