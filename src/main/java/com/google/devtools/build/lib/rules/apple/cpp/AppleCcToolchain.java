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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.XcodeConfig;
import com.google.devtools.build.lib.rules.apple.XcodeConfigInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchain;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables;
import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Implementation for apple_cc_toolchain rule.
 */
public class AppleCcToolchain extends CcToolchain {
  private static final String XCODE_VERSION_KEY = "xcode_version";
  private static final String IOS_SDK_VERSION_KEY = "ios_sdk_version";
  private static final String MACOS_SDK_VERSION_KEY = "macos_sdk_version";
  private static final String TVOS_SDK_VERSION_KEY = "tvos_sdk_version";
  private static final String WATCHOS_SDK_VERSION_KEY = "watchos_sdk_version";
  public static final String SDK_DIR_KEY = "sdk_dir";
  public static final String SDK_FRAMEWORK_DIR_KEY = "sdk_framework_dir";
  public static final String PLATFORM_DEVELOPER_FRAMEWORK_DIR = "platform_developer_framework_dir";
  public static final String VERSION_MIN_KEY = "version_min";

  @VisibleForTesting
  public static final String XCODE_VERSION_OVERRIDE_VALUE_KEY = "xcode_version_override_value";

  @VisibleForTesting
  public static final String APPLE_SDK_VERSION_OVERRIDE_VALUE_KEY =
      "apple_sdk_version_override_value";

  @VisibleForTesting
  public static final String APPLE_SDK_PLATFORM_VALUE_KEY = "apple_sdk_platform_value";

  @Override
  protected AdditionalBuildVariablesComputer getAdditionalBuildVariablesComputer(
      RuleContext ruleContextPossiblyInHostConfiguration) {
    // xcode config is shared between target and host configuration therefore we can use it.
    XcodeConfigInfo xcodeConfig =
        XcodeConfig.getXcodeConfigInfo(ruleContextPossiblyInHostConfiguration);
    return getAdditionalBuildVariablesComputer(xcodeConfig);
  }

  /** Returns {@link AdditionalBuildVariablesComputer} lambda without capturing instance state. */
  private static AdditionalBuildVariablesComputer getAdditionalBuildVariablesComputer(
      XcodeConfigInfo xcodeConfig) {
    return (AdditionalBuildVariablesComputer & Serializable)
        (BuildOptions buildOptions) -> computeCcToolchainVariables(xcodeConfig, buildOptions);
  }

  private static CcToolchainVariables computeCcToolchainVariables(
      XcodeConfigInfo xcodeConfig, BuildOptions buildOptions) {
    AppleConfiguration appleConfiguration = new AppleConfiguration(buildOptions);
    ApplePlatform platform = appleConfiguration.getSingleArchPlatform();
    String cpu = buildOptions.get(CoreOptions.class).cpu;

    Map<String, String> appleEnv = getEnvironmentBuildVariables(xcodeConfig, cpu);
    CcToolchainVariables.Builder variables = CcToolchainVariables.builder();
    variables
        .addStringVariable(
            XCODE_VERSION_KEY, xcodeConfig.getXcodeVersion().toStringWithMinimumComponents(2))
        .addStringVariable(
            IOS_SDK_VERSION_KEY,
            xcodeConfig
                .getSdkVersionForPlatform(ApplePlatform.IOS_SIMULATOR)
                .toStringWithMinimumComponents(2))
        .addStringVariable(
            MACOS_SDK_VERSION_KEY,
            xcodeConfig
                .getSdkVersionForPlatform(ApplePlatform.MACOS)
                .toStringWithMinimumComponents(2))
        .addStringVariable(
            TVOS_SDK_VERSION_KEY,
            xcodeConfig
                .getSdkVersionForPlatform(ApplePlatform.TVOS_SIMULATOR)
                .toStringWithMinimumComponents(2))
        .addStringVariable(
            WATCHOS_SDK_VERSION_KEY,
            xcodeConfig
                .getSdkVersionForPlatform(ApplePlatform.WATCHOS_SIMULATOR)
                .toStringWithMinimumComponents(2))
        .addStringVariable(SDK_DIR_KEY, AppleToolchain.sdkDir())
        .addStringVariable(
            SDK_FRAMEWORK_DIR_KEY, AppleToolchain.sdkFrameworkDir(platform, xcodeConfig))
        .addStringVariable(
            PLATFORM_DEVELOPER_FRAMEWORK_DIR,
            AppleToolchain.platformDeveloperFrameworkDir(platform))
        .addStringVariable(
            XCODE_VERSION_OVERRIDE_VALUE_KEY,
            appleEnv.getOrDefault(AppleConfiguration.XCODE_VERSION_ENV_NAME, ""))
        .addStringVariable(
            APPLE_SDK_VERSION_OVERRIDE_VALUE_KEY,
            appleEnv.getOrDefault(AppleConfiguration.APPLE_SDK_VERSION_ENV_NAME, ""))
        .addStringVariable(
            APPLE_SDK_PLATFORM_VALUE_KEY,
            appleEnv.getOrDefault(AppleConfiguration.APPLE_SDK_PLATFORM_ENV_NAME, ""))
        .addStringVariable(
            VERSION_MIN_KEY,
            xcodeConfig.getMinimumOsForPlatformType(platform.getType()).toString());
    return variables.build();
  }

  private static ImmutableMap<String, String> getEnvironmentBuildVariables(
      XcodeConfigInfo xcodeConfig, String cpu) {
    Map<String, String> builder = new LinkedHashMap<>();
    builder.putAll(AppleConfiguration.getXcodeVersionEnv(xcodeConfig.getXcodeVersion()));
    if (ApplePlatform.isApplePlatform(cpu)) {
      ApplePlatform platform = ApplePlatform.forTargetCpu(cpu);
      builder.putAll(
          AppleConfiguration.appleTargetPlatformEnv(
              platform, xcodeConfig.getSdkVersionForPlatform(platform)));
    }
    return ImmutableMap.copyOf(builder);
  }

  @Override
  protected boolean isAppleToolchain() {
    return true;
  }

  @Override
  protected void validateToolchain(RuleContext ruleContext) throws RuleErrorException {
    if (XcodeConfig.getXcodeConfigInfo(ruleContext).getXcodeVersion() == null) {
      ruleContext.throwWithRuleError(
          "Xcode version must be specified to use an Apple CROSSTOOL. If your Xcode version has "
              + "changed recently, verify that \"xcode-select -p\" is correct and then try: "
              + "\"bazel shutdown\" to re-run Xcode configuration");
    }
  }
}
