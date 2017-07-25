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
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import javax.annotation.Nullable;

/**
 * The set of Apple versions computed from command line options and the {@code xcode_config} rule.
 */
@Immutable
public class XcodeConfigProvider implements TransitiveInfoProvider {
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

  public DottedVersion getXcodeVersion() {
    return xcodeVersion;
  }

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
