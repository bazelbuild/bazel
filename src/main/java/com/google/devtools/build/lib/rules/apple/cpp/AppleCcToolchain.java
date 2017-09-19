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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.XcodeConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchain;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
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
  public static final String XCODE_VERISON_OVERRIDE_VALUE_KEY = "xcode_version_override_value";
  
  @VisibleForTesting
  public static final String APPLE_SDK_VERSION_OVERRIDE_VALUE_KEY =
      "apple_sdk_version_override_value";
  
  @VisibleForTesting
  public static final String APPLE_SDK_PLATFORM_VALUE_KEY = "apple_sdk_platform_value";

  @Override
  protected void addBuildVariables(RuleContext ruleContext, Variables.Builder variables)
      throws RuleErrorException {
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    if (XcodeConfig.getXcodeVersion(ruleContext) == null) {
      ruleContext.throwWithRuleError("Xcode version must be specified to use an Apple CROSSTOOL");
    }

    ApplePlatform platform = appleConfiguration.getSingleArchPlatform();

    Map<String, String> appleEnv = getEnvironmentBuildVariables(ruleContext);

    variables
        .addStringVariable(
            XCODE_VERSION_KEY,
            XcodeConfig.getXcodeVersion(ruleContext).toStringWithMinimumComponents(2))
        .addStringVariable(
            IOS_SDK_VERSION_KEY,
            XcodeConfig.getSdkVersionForPlatform(ruleContext, ApplePlatform.IOS_SIMULATOR)
                .toStringWithMinimumComponents(2))
        .addStringVariable(
            MACOS_SDK_VERSION_KEY,
            XcodeConfig.getSdkVersionForPlatform(ruleContext, ApplePlatform.MACOS)
                .toStringWithMinimumComponents(2))
        .addStringVariable(
            TVOS_SDK_VERSION_KEY,
            XcodeConfig.getSdkVersionForPlatform(ruleContext, ApplePlatform.TVOS_SIMULATOR)
                .toStringWithMinimumComponents(2))
        .addStringVariable(
            WATCHOS_SDK_VERSION_KEY,
            XcodeConfig.getSdkVersionForPlatform(ruleContext, ApplePlatform.WATCHOS_SIMULATOR)
                .toStringWithMinimumComponents(2))
        .addStringVariable(SDK_DIR_KEY, AppleToolchain.sdkDir())
        .addStringVariable(
            SDK_FRAMEWORK_DIR_KEY, AppleToolchain.sdkFrameworkDir(platform, ruleContext))
        .addStringVariable(
            PLATFORM_DEVELOPER_FRAMEWORK_DIR,
            AppleToolchain.platformDeveloperFrameworkDir(appleConfiguration))
        .addStringVariable(
            XCODE_VERISON_OVERRIDE_VALUE_KEY,
            appleEnv.getOrDefault(AppleConfiguration.XCODE_VERSION_ENV_NAME, ""))
        .addStringVariable(
            APPLE_SDK_VERSION_OVERRIDE_VALUE_KEY,
            appleEnv.getOrDefault(AppleConfiguration.APPLE_SDK_VERSION_ENV_NAME, ""))
        .addStringVariable(
            APPLE_SDK_PLATFORM_VALUE_KEY,
            appleEnv.getOrDefault(AppleConfiguration.APPLE_SDK_PLATFORM_ENV_NAME, ""))
        .addStringVariable(
            VERSION_MIN_KEY,
            XcodeConfig.getMinimumOsForPlatformType(ruleContext, platform.getType()).toString());
  }

  @Override
  protected NestedSet<Artifact> fullInputsForLink(
      RuleContext ruleContext, NestedSet<Artifact> link) {
    return NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(link)
        .addTransitive(AnalysisUtils.getMiddlemanFor(ruleContext, ":libc_top", Mode.TARGET))
        .build();
  }

  private ImmutableMap<String, String> getEnvironmentBuildVariables(RuleContext ruleContext) {
    Map<String, String> builder = new LinkedHashMap<>();
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    builder.putAll(appleConfiguration.getAppleHostSystemEnv());
    if (ApplePlatform.isApplePlatform(cppConfiguration.getTargetCpu())) {
      builder.putAll(
          appleConfiguration.appleTargetPlatformEnv(
              ApplePlatform.forTargetCpu(cppConfiguration.getTargetCpu())));
    }
    return ImmutableMap.copyOf(builder);
  }
}
