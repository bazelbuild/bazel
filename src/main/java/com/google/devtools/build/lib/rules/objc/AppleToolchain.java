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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.starlarkbuildapi.apple.AppleToolchainApi;

/**
 * Utility class for resolving items for the Apple toolchain (such as common tool flags, and paths).
 */
@Immutable
public class AppleToolchain implements AppleToolchainApi<AppleConfiguration> {

  // These next two strings are shared secrets with the xcrunwrapper.sh to allow
  // expansion of DeveloperDir and SDKRoot and runtime, since they aren't known
  // until compile time on any given build machine.
  private static final String DEVELOPER_DIR = "__BAZEL_XCODE_DEVELOPER_DIR__";
  private static final String SDKROOT_DIR = "__BAZEL_XCODE_SDKROOT__";

  // These two paths are framework paths relative to SDKROOT.
  @VisibleForTesting
  public static final String DEVELOPER_FRAMEWORK_PATH = "/Developer/Library/Frameworks";
  @VisibleForTesting
  public static final String SYSTEM_FRAMEWORK_PATH = "/System/Library/Frameworks";

  /** Returns the platform directory inside of Xcode for a platform name. */
  public static String platformDir(String platformName) {
    return developerDir() + "/Platforms/" + platformName + ".platform";
  }

  /**
   * Returns the platform directory inside of Xcode for a given configuration.
   */
  public static String sdkDir() {
    return SDKROOT_DIR;
  }

  /**
   * Returns the Developer directory inside of Xcode for a given configuration.
   */
  public static String developerDir() {
    return DEVELOPER_DIR;
  }

  /**
   * Returns the platform frameworks directory inside of Xcode for a given {@link ApplePlatform}.
   */
  public static String platformDeveloperFrameworkDir(ApplePlatform platform) {
    String platformDir = platformDir(platform.getNameInPlist());
    return platformDir + "/Developer/Library/Frameworks";
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  /**
   * Returns the platform directory inside of Xcode for a given configuration.
   */
  @Override
  public String sdkDirConstant() {
    return sdkDir();
  }

  /**
   * Returns the Developer directory inside of Xcode for a given configuration.
   */
  @Override
  public String developerDirConstant() {
    return developerDir();
  }

  /**
   * Returns the platform frameworks directory inside of Xcode for a given configuration.
   */
  @Override
  public String platformFrameworkDirFromConfig(AppleConfiguration configuration) {
    return platformDeveloperFrameworkDir(configuration.getSingleArchPlatform());
  }
}
