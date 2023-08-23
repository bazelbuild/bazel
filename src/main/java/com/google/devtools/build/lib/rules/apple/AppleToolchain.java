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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.starlarkbuildapi.apple.AppleToolchainApi;
import java.io.Serializable;

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

  /** Returns the SDK frameworks directory inside of Xcode for a given configuration. */
  public static String sdkFrameworkDir(ApplePlatform targetPlatform, XcodeConfigInfo xcodeConfig) {
    String relativePath;
    switch (targetPlatform) {
      case IOS_DEVICE:
      case IOS_SIMULATOR:
        if (xcodeConfig
                .getSdkVersionForPlatform(targetPlatform)
                .compareTo(DottedVersion.fromStringUnchecked("9.0"))
            >= 0) {
          relativePath = SYSTEM_FRAMEWORK_PATH;
        } else {
          relativePath = DEVELOPER_FRAMEWORK_PATH;
        }
        break;
      case MACOS:
      case VISIONOS_DEVICE:
      case VISIONOS_SIMULATOR:
      case WATCHOS_DEVICE:
      case WATCHOS_SIMULATOR:
      case TVOS_DEVICE:
      case TVOS_SIMULATOR:
      case CATALYST:
        relativePath = SYSTEM_FRAMEWORK_PATH;
        break;
      default:
        throw new IllegalArgumentException("Unhandled platform " + targetPlatform);
    }
    return sdkDir() + relativePath;
  }

  /** The default label of the build-wide {@code xcode_config} configuration rule. */
  public static LabelLateBoundDefault<AppleConfiguration> getXcodeConfigLabel(
      RepositoryName toolsRepository) {
    return LabelLateBoundDefault.fromTargetConfiguration(
        AppleConfiguration.class,
        Label.parseCanonicalUnchecked(
            toolsRepository + AppleCommandLineOptions.DEFAULT_XCODE_VERSION_CONFIG_LABEL),
        (Attribute.LateBoundDefault.Resolver<AppleConfiguration, Label> & Serializable)
            (rule, attributes, appleConfig) -> appleConfig.getXcodeConfigLabel());
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

  /**
   * Base rule definition to be ancestor for rules which may require an xcode toolchain.
   */
  public static class RequiresXcodeConfigRule implements RuleDefinition {
    private final RepositoryName toolsRepository;

    public RequiresXcodeConfigRule(RepositoryName toolsRepository) {
      this.toolsRepository = toolsRepository;
    }

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(
              attr(XcodeConfigRule.XCODE_CONFIG_ATTR_NAME, LABEL)
                  .allowedRuleClasses("xcode_config")
                  .checkConstraints()
                  .value(getXcodeConfigLabel(toolsRepository)))
          .build();
    }
    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$requires_xcode_config")
          .type(RuleClassType.ABSTRACT)
          .build();
    }
  }
}
