// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.apple.cpp;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.cpp.CcToolchain;

import java.util.Map;

/**
 * Implementation for apple_cc_toolchain rule.
 */
public class AppleCcToolchain extends CcToolchain {

  // TODO(bazel-team): Compute default based on local Xcode instead of hardcoded 7.2.
  private static final DottedVersion DEFAULT_XCODE_VERSION = DottedVersion.fromString("7.2");

  private static final String XCODE_VERSION_KEY = "xcode_version";
  private static final String IOS_SDK_VERSION_KEY = "ios_sdk_version";
  private static final String MACOSX_SDK_VERSION_KEY = "macosx_sdk_version";
  private static final String TVOS_SDK_VERSION_KEY = "appletvos_sdk_version";
  private static final String WATCHOS_SDK_VERSION_KEY = "watchos_sdk_version";
  public static final String SDK_DIR_KEY = "sdk_dir";
  public static final String SDK_FRAMEWORK_DIR_KEY = "sdk_framework_dir";
  public static final String PLATFORM_DEVELOPER_FRAMEWORK_DIR = "platform_developer_framework_dir";

  @Override
  protected Map<String, String> getBuildVariables(RuleContext ruleContext) {
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    Platform platform = appleConfiguration.getSingleArchPlatform();

    return ImmutableMap.<String, String>builder()
        .put(
            XCODE_VERSION_KEY,
            appleConfiguration.getXcodeVersion().or(DEFAULT_XCODE_VERSION).toString())
        .put(
            IOS_SDK_VERSION_KEY,
            appleConfiguration.getSdkVersionForPlatform(Platform.IOS_SIMULATOR).toString())
        .put(
            MACOSX_SDK_VERSION_KEY,
            appleConfiguration.getSdkVersionForPlatform(Platform.MACOS_X).toString())
        .put(
            TVOS_SDK_VERSION_KEY,
            appleConfiguration.getSdkVersionForPlatform(Platform.TVOS_SIMULATOR).toString())
        .put(
            WATCHOS_SDK_VERSION_KEY,
            appleConfiguration.getSdkVersionForPlatform(Platform.WATCHOS_SIMULATOR).toString())
        .put(SDK_DIR_KEY, AppleToolchain.sdkDir())
        .put(SDK_FRAMEWORK_DIR_KEY, AppleToolchain.sdkFrameworkDir(platform, appleConfiguration))
        .put(
            PLATFORM_DEVELOPER_FRAMEWORK_DIR,
            AppleToolchain.platformDeveloperFrameworkDir(appleConfiguration))
        .build();
  }
}
