// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.apple;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.XcodeprojBuildSetting;

/**
 * Utility class for resolving items for the Apple toolchain (such as common tool flags, and paths).
 */
public class AppleToolchain {
  // These next two strings are shared secrets with the xcrunwrapper.sh to allow
  // expansion of DeveloperDir and SDKRoot and runtime, since they aren't known
  // until compile time on any given build machine.
  private static final String DEVELOPER_DIR = "__BAZEL_XCODE_DEVELOPER_DIR__";
  private static final String SDKROOT_DIR = "__BAZEL_XCODE_SDKROOT__";
  
  // There is a handy reference to many clang warning flags at
  // http://nshipster.com/clang-diagnostics/
  // There is also a useful narrative for many Xcode settings at
  // http://www.xs-labs.com/en/blog/2011/02/04/xcode-build-settings/
  public static final ImmutableMap<String, String> DEFAULT_WARNINGS =
      new ImmutableMap.Builder<String, String>()
          .put("GCC_WARN_64_TO_32_BIT_CONVERSION", "-Wshorten-64-to-32")
          .put("CLANG_WARN_BOOL_CONVERSION", "-Wbool-conversion")
          .put("CLANG_WARN_CONSTANT_CONVERSION", "-Wconstant-conversion")
          // Double-underscores are intentional - thanks Xcode.
          .put("CLANG_WARN__DUPLICATE_METHOD_MATCH", "-Wduplicate-method-match")
          .put("CLANG_WARN_EMPTY_BODY", "-Wempty-body")
          .put("CLANG_WARN_ENUM_CONVERSION", "-Wenum-conversion")
          .put("CLANG_WARN_INT_CONVERSION", "-Wint-conversion")
          .put("CLANG_WARN_UNREACHABLE_CODE", "-Wunreachable-code")
          .put("GCC_WARN_ABOUT_RETURN_TYPE", "-Wmismatched-return-types")
          .put("GCC_WARN_UNDECLARED_SELECTOR", "-Wundeclared-selector")
          .put("GCC_WARN_UNINITIALIZED_AUTOS", "-Wuninitialized")
          .put("GCC_WARN_UNUSED_FUNCTION", "-Wunused-function")
          .put("GCC_WARN_UNUSED_VARIABLE", "-Wunused-variable")
          .build();

  private AppleToolchain() {
    throw new UnsupportedOperationException("static-only");
  }

  /**
   * Returns the platform plist name (for example, iPhoneSimulator) for the platform corresponding
   * to the value of {@code --ios_cpu} in the given configuration.
   */
  // TODO(bazel-team): Support non-ios platforms.
  public static String getPlatformPlistName(AppleConfiguration configuration) {
    return Platform.forIosArch(configuration.getIosCpu()).getNameInPlist();
  }

  /**
   * Returns the platform directory inside of Xcode for a given configuration.
   */
  public static String platformDir(AppleConfiguration configuration) {
    return platformDir(getPlatformPlistName(configuration));
  }

  /**
   * Returns the platform directory inside of Xcode for a given platform name (e.g. iphoneos).
   */
  public static String platformDir(String platformName) {
    return DEVELOPER_DIR + "/Platforms/" + platformName + ".platform";
  }

  /**
   * Returns the platform directory inside of Xcode for a given configuration.
   */
  public static String sdkDir() {
    return SDKROOT_DIR;
  }

  /**
   * Returns the platform frameworks directory inside of Xcode for a given configuration.
   */
  public static String platformDeveloperFrameworkDir(AppleConfiguration configuration) {
    return platformDir(configuration) + "/Developer/Library/Frameworks";
  }

  /**
   * Returns the SDK frameworks directory inside of Xcode for a given configuration.
   */
  public static String sdkDeveloperFrameworkDir() {
    return sdkDir() + "/Developer/Library/Frameworks";
  }

  /**
   * Returns swift libraries path.
   */
  public static String swiftLibDir(AppleConfiguration configuration) {
    return DEVELOPER_DIR + "/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/"
        + swiftPlatform(configuration);
  }

  /**
   * Returns a platform name string suitable for use in Swift tools.
   */
  public static String swiftPlatform(AppleConfiguration configuration) {
    return getPlatformPlistName(configuration).toLowerCase();
  }

  /**
   * Returns a series of xcode build settings which configure compilation warnings to
   * "recommended settings". Without these settings, compilation might result in some spurious
   * warnings, and xcode would complain that the settings be changed to these values.
   */
  public static Iterable<? extends XcodeprojBuildSetting> defaultWarningsForXcode() {
    return Iterables.transform(DEFAULT_WARNINGS.keySet(),
        new Function<String, XcodeprojBuildSetting>() {
      @Override
      public XcodeprojBuildSetting apply(String key) {
        return XcodeprojBuildSetting.newBuilder().setName(key).setValue("YES").build();
      }
    });
  }
}
