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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.Locale;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * An enum that can be used to distinguish between various apple platforms.
 */
public enum Platform {
  IOS_DEVICE("iPhoneOS"),
  IOS_SIMULATOR("iPhoneSimulator"),
  MACOS_X("MacOSX"),
  TVOS_DEVICE("AppleTVOS"),
  TVOS_SIMULATOR("AppleTVSimulator"),
  WATCHOS_DEVICE("WatchOS"),
  WATCHOS_SIMULATOR("WatchSimulator");

  private static final Set<String> IOS_SIMULATOR_ARCHS = ImmutableSet.of("i386", "x86_64");
  private static final Set<String> IOS_DEVICE_ARCHS =
      ImmutableSet.of("armv6", "armv7", "armv7s", "arm64");
  
  private static final Set<String> IOS_SIMULATOR_TARGET_CPUS =
      ImmutableSet.of("ios_x86_64", "ios_i386");
  private static final Set<String> IOS_DEVICE_TARGET_CPUS =
      ImmutableSet.of("ios_armv7", "ios_arm64");
  private static final Set<String> MACOSX_TARGET_CPUS =
      ImmutableSet.of("darwin_x86_64");

  private final String nameInPlist;

  Platform(String nameInPlist) {
    this.nameInPlist = Preconditions.checkNotNull(nameInPlist);
  }

  /**
   * Returns the name of the "platform" as it appears in the CFBundleSupportedPlatforms plist
   * setting.
   */
  public String getNameInPlist() {
    return nameInPlist;
  }

  /**
   * Returns the name of the "platform" as it appears in the plist when it appears in all-lowercase.
   */
  public String getLowerCaseNameInPlist() {
    return nameInPlist.toLowerCase(Locale.US);
  }

  /**
   * Returns the iOS platform for the given iOS architecture.
   *
   * <p>If this method is used in non-iOS contexts, results are undefined. If the input happens
   * to share an architecture with some iOS platform, this will return that platform even if it is
   * incorrect (for example, IOS_SIMULATOR for the x86_64 of darwin_x86_64).
   * 
   * @throws IllegalArgumentException if there is no valid ios platform for the given architecture
   */
  public static Platform forIosArch(String arch) {
    if (IOS_SIMULATOR_ARCHS.contains(arch)) {
      return IOS_SIMULATOR;
    } else if (IOS_DEVICE_ARCHS.contains(arch)) {
      return IOS_DEVICE;
    } else {
      throw new IllegalArgumentException(
          "No supported ios platform registered for architecture " + arch);
    }
  }
  
  @Nullable
  private static Platform forTargetCpuNullable(String targetCpu) {
    if (IOS_SIMULATOR_TARGET_CPUS.contains(targetCpu)) {
      return IOS_SIMULATOR;
    } else if (IOS_DEVICE_TARGET_CPUS.contains(targetCpu)) {
      return IOS_DEVICE;
    } else if (MACOSX_TARGET_CPUS.contains(targetCpu)) {
      return MACOS_X;
    } else {
      return null;
    }
  }

  /**
   * Returns the platform for the given target cpu.
   * 
   * @throws IllegalArgumentException if there is no valid apple platform for the given target cpu
   */
  public static Platform forTargetCpu(String targetCpu) {
    Platform platform = forTargetCpuNullable(targetCpu);
    if (platform != null) {
      return platform; 
    } else {
      throw new IllegalArgumentException(
          "No supported apple platform registered for target cpu " + targetCpu);
    }
  }
  
  /**
   * Returns true if the given target cpu is an apple platform.
   */
  public static boolean isApplePlatform(String targetCpu) {
    return forTargetCpuNullable(targetCpu) != null;
  }
}
