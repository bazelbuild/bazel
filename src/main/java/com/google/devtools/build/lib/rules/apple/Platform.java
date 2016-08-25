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
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.Locale;
import java.util.Set;
import javax.annotation.Nullable;

/** An enum that can be used to distinguish between various apple platforms. */
@SkylarkModule(
  name = "platform",
  category = SkylarkModuleCategory.NONE,
  doc = "Distinguishes between various apple platforms."
)
public enum Platform {

  IOS_DEVICE("iPhoneOS", PlatformType.IOS),
  IOS_SIMULATOR("iPhoneSimulator", PlatformType.IOS),
  MACOS_X("MacOSX", PlatformType.MACOSX),
  TVOS_DEVICE("AppleTVOS", PlatformType.TVOS),
  TVOS_SIMULATOR("AppleTVSimulator", PlatformType.TVOS),
  WATCHOS_DEVICE("WatchOS", PlatformType.WATCHOS),
  WATCHOS_SIMULATOR("WatchSimulator", PlatformType.WATCHOS);

  private static final Set<String> IOS_SIMULATOR_TARGET_CPUS =
      ImmutableSet.of("ios_x86_64", "ios_i386");
  private static final Set<String> WATCHOS_SIMULATOR_TARGET_CPUS =
      ImmutableSet.of("watchos_i386");
  private static final Set<String> WATCHOS_DEVICE_TARGET_CPUS =
      ImmutableSet.of("watchos_armv7k");
  private static final Set<String> IOS_DEVICE_TARGET_CPUS =
      ImmutableSet.of("ios_armv6", "ios_arm64", "ios_armv7", "ios_armv7s");
  private static final Set<String> MACOSX_TARGET_CPUS =
      ImmutableSet.of("darwin_x86_64");

  private final String nameInPlist;
  private final PlatformType platformType;

  Platform(String nameInPlist, PlatformType platformType) {
    this.nameInPlist = Preconditions.checkNotNull(nameInPlist);
    this.platformType = platformType;
  }

  /** Returns the platform type of this platform. */
  @SkylarkCallable(
    name = "platform_type",
    doc = "Returns the platform type of this platform.",
    structField = true
  )
  public PlatformType getType() {
    return platformType;
  }

  /**
   * Returns the name of the "platform" as it appears in the CFBundleSupportedPlatforms plist
   * setting.
   */
  @SkylarkCallable(name = "name_in_plist", structField = true,
    doc = "The name of the platform as it appears in the CFBundleSupportedPlatforms plist "
        + "setting. This name can also be converted to lowercase and passed to command-line "
        + "tools, such as ibtool and actool.")
  public String getNameInPlist() {
    return nameInPlist;
  }

  /**
   * Returns the name of the "platform" as it appears in the plist when it appears in all-lowercase.
   */
  public String getLowerCaseNameInPlist() {
    return nameInPlist.toLowerCase(Locale.US);
  }

  @Nullable
  private static Platform forTargetCpuNullable(String targetCpu) {
    if (IOS_SIMULATOR_TARGET_CPUS.contains(targetCpu)) {
      return IOS_SIMULATOR;
    } else if (IOS_DEVICE_TARGET_CPUS.contains(targetCpu)) {
      return IOS_DEVICE;
    } else if (WATCHOS_SIMULATOR_TARGET_CPUS.contains(targetCpu)) {
      return WATCHOS_SIMULATOR;
    } else if (WATCHOS_DEVICE_TARGET_CPUS.contains(targetCpu)) {
      return WATCHOS_DEVICE;
    } else if (MACOSX_TARGET_CPUS.contains(targetCpu)) {
      return MACOS_X;
    } else {
      return null;
    }
  }

  /**
   * Returns the platform for the given target cpu and platform type.
   * 
   * @param platformType platform type that the given cpu value is implied for
   * @param arch architecture representation, such as 'arm64'
   * @throws IllegalArgumentException if there is no valid apple platform for the given target cpu
   */
  public static Platform forTarget(PlatformType platformType, String arch) {
    return forTargetCpu(String.format("%s_%s", platformType.toString(), arch));
  }

 /**
  * Returns the platform for the given target cpu.
  * 
  * @param targetCpu cpu value with platform type prefix, such as 'ios_arm64'
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

  /**
   * Value used to describe Apple platform "type". A {@link Platform} is implied from a platform
   * type (for example, watchOS) together with a cpu value (for example, armv7).
   */
  // TODO(cparsons): Use these values in static retrieval methods in this class.
  @SkylarkModule(
    name = "platform_type",
    category = SkylarkModuleCategory.NONE,
    doc = "Describes Apple platform \"type\", such as iOS, tvOS, macOS etc."
  )
  public enum PlatformType {
    IOS,
    WATCHOS,
    TVOS,
    MACOSX;
    
    @Override
    public String toString() {
      return name().toLowerCase();
    }
    
    /**
     * Returns the {@link PlatformType} with given name (case insensitive).
     * 
     * @throws IllegalArgumentException if the name does not match a valid platform type.
     */
    public static PlatformType fromString(String name) {
      for (PlatformType platformType : PlatformType.values()) {
        if (name.equalsIgnoreCase(platformType.toString())) {
          return platformType;
        }
      }
      throw new IllegalArgumentException(String.format("Unsupported platform type \"%s\"", name));
    }
  }
}
