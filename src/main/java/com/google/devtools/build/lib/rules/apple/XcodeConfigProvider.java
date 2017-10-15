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
package com.google.devtools.build.lib.rules.apple;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import javax.annotation.Nullable;

/**
 * The set of Apple versions computed from command line options and the {@code xcode_config} rule.
 */
@Immutable
@SkylarkModule(
    name = XcodeConfigProvider.SKYLARK_NAME,
    doc = "The set of Apple versions computed from command line options and the xcode_config rule.")
public class XcodeConfigProvider extends NativeInfo {
  /** Skylark name for this provider. */
  public static final String SKYLARK_NAME = "XcodeVersionConfig";

  /** Provider identifier for {@link XcodeConfigProvider}. */
  public static final NativeProvider<XcodeConfigProvider> PROVIDER =
      new NativeProvider<XcodeConfigProvider>(XcodeConfigProvider.class, SKYLARK_NAME) {};

  private final DottedVersion iosSdkVersion;
  private final DottedVersion iosMinimumOsVersion;
  private final DottedVersion watchosSdkVersion;
  private final DottedVersion watchosMinimumOsVersion;
  private final DottedVersion tvosSdkVersion;
  private final DottedVersion tvosMinimumOsVersion;
  private final DottedVersion macosSdkVersion;
  private final DottedVersion macosMinimumOsVersion;
  @Nullable private final DottedVersion xcodeVersion;

  public XcodeConfigProvider(
      DottedVersion iosSdkVersion, DottedVersion iosMinimumOsVersion,
      DottedVersion watchosSdkVersion, DottedVersion watchosMinimumOsVersion,
      DottedVersion tvosSdkVersion, DottedVersion tvosMinimumOsVersion,
      DottedVersion macosSdkVersion, DottedVersion macosMinimumOsVersion,
      DottedVersion xcodeVersion) {
    super(PROVIDER, ImmutableMap.of());
    this.iosSdkVersion = Preconditions.checkNotNull(iosSdkVersion);
    this.iosMinimumOsVersion = Preconditions.checkNotNull(iosMinimumOsVersion);
    this.watchosSdkVersion = Preconditions.checkNotNull(watchosSdkVersion);
    this.watchosMinimumOsVersion = Preconditions.checkNotNull(watchosMinimumOsVersion);
    this.tvosSdkVersion = Preconditions.checkNotNull(tvosSdkVersion);
    this.tvosMinimumOsVersion = Preconditions.checkNotNull(tvosMinimumOsVersion);
    this.macosSdkVersion = Preconditions.checkNotNull(macosSdkVersion);
    this.macosMinimumOsVersion = Preconditions.checkNotNull(macosMinimumOsVersion);
    this.xcodeVersion = xcodeVersion;
  }

  @SkylarkCallable(name = "xcode_version",
      doc = "Returns the Xcode version that is being used to build.<p>"
          + "This will return <code>None</code> if no Xcode versions are available.",
      allowReturnNones = true)
  public DottedVersion getXcodeVersion() {
    return xcodeVersion;
  }

  @SkylarkCallable(
      name = "minimum_os_for_platform_type",
      doc = "The minimum compatible OS version for target simulator and devices for a particular "
          + "platform type.")
  public DottedVersion getMinimumOsForPlatformType(ApplePlatform.PlatformType platformType) {
    // TODO(b/37240784): Look into using only a single minimum OS flag tied to the current
    // apple_platform_type.
    switch (platformType) {
      case IOS:
        return iosMinimumOsVersion;
      case TVOS:
        return tvosMinimumOsVersion;
      case WATCHOS:
        return watchosMinimumOsVersion;
      case MACOS:
        return macosMinimumOsVersion;
      default:
        throw new IllegalArgumentException("Unhandled platform type: " + platformType);
    }
  }

  @SkylarkCallable(
      name = "sdk_version_for_platform",
      doc = "The version of the platform SDK that will be used to build targets for the given "
          + "platform.")
  public DottedVersion getSdkVersionForPlatform(ApplePlatform platform) {
    switch (platform) {
      case IOS_DEVICE:
      case IOS_SIMULATOR:
        return iosSdkVersion;
      case TVOS_DEVICE:
      case TVOS_SIMULATOR:
        return tvosSdkVersion;
      case WATCHOS_DEVICE:
      case WATCHOS_SIMULATOR:
        return watchosSdkVersion;
      case MACOS:
        return macosSdkVersion;
      default:
        throw new IllegalArgumentException("Unhandled platform: " + platform);
    }
  }
}
