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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.ROOT_MERGE_ZIP;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.apple.Platform.PlatformType;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.XcodeprojBuildSetting;

/**
 * Contains support methods for common processing and generating of watch extension and application
 * bundles.
 */
// TODO(b/30503590): Refactor this into a support class -- such classes are better than this static
// utility.
final class WatchUtils {

  @VisibleForTesting
  /** Bundle directory format for watch applications for watch OS 2. */
  static final String WATCH2_APP_BUNDLE_DIR_FORMAT = "Watch/%s.app";

  /**
   * Supported Apple watch OS versions.
   */
  enum WatchOSVersion {
    OS1(
        XcodeProductType.WATCH_OS1_APPLICATION,
        XcodeProductType.WATCH_OS1_EXTENSION,
        ReleaseBundlingSupport.APP_BUNDLE_DIR_FORMAT,
        "WatchKitSupport"),
    OS2(
        XcodeProductType.WATCH_OS2_APPLICATION,
        XcodeProductType.WATCH_OS2_EXTENSION,
        WATCH2_APP_BUNDLE_DIR_FORMAT,
        "WatchKitSupport2");

    private final XcodeProductType applicationXcodeProductType;
    private final XcodeProductType extensionXcodeProductType;
    private final String applicationBundleDirFormat;
    private final String watchKitSupportDirName;

    WatchOSVersion(
        XcodeProductType applicationXcodeProductType,
        XcodeProductType extensionXcodeProductType,
        String applicationBundleDirFormat,
        String watchKitSupportDirName) {
      this.applicationXcodeProductType = applicationXcodeProductType;
      this.extensionXcodeProductType = extensionXcodeProductType;
      this.applicationBundleDirFormat = applicationBundleDirFormat;
      this.watchKitSupportDirName = watchKitSupportDirName;
    }
    
    /**
     * Returns the {@link XcodeProductType} to be used for the watch application's Xcode target.
     */
    XcodeProductType getApplicationXcodeProductType() {
      return applicationXcodeProductType;
    }

    /**
     * Returns the {@link XcodeProductType} to be used for the watch extension's Xcode target.
     */
    XcodeProductType getExtensionXcodeProductType() {
      return extensionXcodeProductType;
    }

    /** Returns the bundle directory format of the watch application relative to its container. */
    String getApplicationBundleDirFormat() {
      return applicationBundleDirFormat;
    }

    /**
     * Returns the name of the directory in the final iOS bundle which should contain the WatchKit
     * support stub.
     */
    String getWatchKitSupportDirName() {
      return watchKitSupportDirName;
    }
  }

  @VisibleForTesting
  static final String WATCH_KIT_STUB_PATH =
      "${SDKROOT}/Library/Application\\ Support/WatchKit/WK";

  // Apple only accepts watch extension and application starting at 8.2.
  @VisibleForTesting
  static final DottedVersion MINIMUM_OS_VERSION = DottedVersion.fromString("8.2");

  /**
   * Adds common xcode build settings required for watch bundles to the given xcode provider
   * builder.
   */
  static void addXcodeSettings(RuleContext ruleContext,
      XcodeProvider.Builder xcodeProviderBuilder) {
    xcodeProviderBuilder.addMainTargetXcodeprojBuildSettings(xcodeSettings(ruleContext));
  }

  /**
   * Watch Extension are not accepted by Apple below iOS version 8.2. While applications built with
   * a minimum iOS version of less than 8.2 may contain watch extension in their bundle, the
   * extension itself needs to be built with 8.2 or higher. This logic overrides (if necessary)
   * any flag-set minimum iOS version for extensions only so that this requirement is not
   * violated.
   */
  static DottedVersion determineMinimumIosVersion(DottedVersion fromFlag) {
    return Ordering.natural().max(fromFlag, MINIMUM_OS_VERSION);
  }

  static boolean isBuildingForWatchOS1Version(WatchOSVersion watchOSVersion) {
    return watchOSVersion == WatchOSVersion.OS1;
  }
 
  /**
   * Adds WatchKitSupport stub to the final ipa and exposes it to given @{link ObjcProvider.Builder}
   * based on watch OS version.
   *
   * For example, for watch OS 1, the resulting ipa will have:
   *   Payload/TestApp.app
   *   WatchKitSupport
   *   WatchKitSupport/WK
   */
  static void registerActionsToAddWatchSupport(
      RuleContext ruleContext, ObjcProvider.Builder objcProviderBuilder,
      WatchOSVersion watchOSVersion) {
    Artifact watchSupportZip = watchSupportZip(ruleContext);
    String workingDirectory = watchSupportZip.getExecPathString()
        .substring(0, watchSupportZip.getExecPathString().lastIndexOf('/'));
    String watchKitSupportDirName = watchOSVersion.getWatchKitSupportDirName();
    String watchKitSupportPath = workingDirectory + "/" + watchKitSupportDirName;

    ImmutableList<String> command = ImmutableList.of(
        // 1. Copy WK stub binary to watchKitSupportPath.
        "mkdir -p " + watchKitSupportPath,
        "&&",
        String.format("cp -f %s %s", WATCH_KIT_STUB_PATH, watchKitSupportPath),
        // 2. cd to working directory.
        "&&",
        "cd " + workingDirectory,
        // 3. Zip watchSupport.
        "&&",
        String.format(
            "/usr/bin/zip -q -r -0 %s %s",
            watchSupportZip.getFilename(),
            watchKitSupportDirName));

    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    ruleContext.registerAction(
        ObjcRuleClasses.spawnAppleEnvActionBuilder(
                appleConfiguration,
                appleConfiguration.getMultiArchPlatform(PlatformType.WATCHOS))
            .setProgressMessage("Copying Watchkit support to app bundle")
            .setShellCommand(ImmutableList.of("/bin/bash", "-c", Joiner.on(" ").join(command)))
            .addOutput(watchSupportZip)
            .build(ruleContext));

    objcProviderBuilder.add(ROOT_MERGE_ZIP, watchSupportZip(ruleContext));
  }

  private static Artifact watchSupportZip(RuleContext ruleContext) {
    return ruleContext.getRelatedArtifact(
        ruleContext.getUniqueDirectory("_watch"), "/WatchKitSupport.zip");
  }

  private static Iterable<XcodeprojBuildSetting> xcodeSettings(RuleContext ruleContext) {
    ImmutableList.Builder<XcodeprojBuildSetting> xcodeSettings = new ImmutableList.Builder<>();
    xcodeSettings.add(
        XcodeprojBuildSetting.newBuilder()
            .setName("RESOURCES_TARGETED_DEVICE_FAMILY")
            .setValue(TargetDeviceFamily.WATCH.getNameInRule())
            .build());
    xcodeSettings.add(
        XcodeprojBuildSetting.newBuilder()
            .setName("IPHONEOS_DEPLOYMENT_TARGET")
            .setValue(determineMinimumIosVersion(
                ruleContext.getFragment(AppleConfiguration.class)
                    .getMinimumOsForPlatformType(PlatformType.IOS)).toString())
            .build());
    return xcodeSettings.build();
  }
}
