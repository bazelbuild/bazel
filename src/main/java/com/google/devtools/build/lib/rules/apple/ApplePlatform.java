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

import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.starlarkbuildapi.apple.ApplePlatformApi;
import java.util.HashMap;
import java.util.Locale;
import javax.annotation.Nullable;
import net.starlark.java.eval.Printer;
import net.starlark.java.syntax.Location;

// LINT.IfChange
// TODO(b/331163027): Remove this duplicate of common/objc/apple_platform.PLATFORM_TYPE
/**
 * An enum that can be used to distinguish between various apple platforms.
 *
 * <p>The enum Platform is being migrated to a Starlark struct PLATFORM in
 * builtins_bzl/common/objc/apple_platform.bzl.
 */
@Immutable
public enum ApplePlatform implements ApplePlatformApi {
  IOS_DEVICE("ios_device", "iPhoneOS", PlatformType.IOS, true),
  IOS_SIMULATOR("ios_simulator", "iPhoneSimulator", PlatformType.IOS, false),
  MACOS("macos", "MacOSX", PlatformType.MACOS, true),
  TVOS_DEVICE("tvos_device", "AppleTVOS", PlatformType.TVOS, true),
  TVOS_SIMULATOR("tvos_simulator", "AppleTVSimulator", PlatformType.TVOS, false),
  VISIONOS_DEVICE("visionos_device", "XROS", PlatformType.VISIONOS, true),
  VISIONOS_SIMULATOR("visionos_simulator", "XRSimulator", PlatformType.VISIONOS, false),
  WATCHOS_DEVICE("watchos_device", "WatchOS", PlatformType.WATCHOS, true),
  WATCHOS_SIMULATOR("watchos_simulator", "WatchSimulator", PlatformType.WATCHOS, false),
  CATALYST("catalyst", "MacOSX", PlatformType.CATALYST, true);

  private static final ImmutableSet<String> IOS_SIMULATOR_TARGET_CPUS =
      ImmutableSet.of("ios_x86_64", "ios_i386", "ios_sim_arm64");
  private static final ImmutableSet<String> IOS_DEVICE_TARGET_CPUS =
      ImmutableSet.of("ios_armv6", "ios_arm64", "ios_armv7", "ios_armv7s", "ios_arm64e");
  private static final ImmutableSet<String> VISIONOS_SIMULATOR_TARGET_CPUS =
      ImmutableSet.of("visionos_sim_arm64");
  private static final ImmutableSet<String> VISIONOS_DEVICE_TARGET_CPUS =
      ImmutableSet.of("visionos_arm64");
  private static final ImmutableSet<String> WATCHOS_SIMULATOR_TARGET_CPUS =
      ImmutableSet.of("watchos_i386", "watchos_x86_64", "watchos_arm64");
  private static final ImmutableSet<String> WATCHOS_DEVICE_TARGET_CPUS =
      ImmutableSet.of(
          "watchos_armv7k", "watchos_arm64_32", "watchos_device_arm64", "watchos_device_arm64e");
  private static final ImmutableSet<String> TVOS_SIMULATOR_TARGET_CPUS =
      ImmutableSet.of("tvos_x86_64", "tvos_sim_arm64");
  private static final ImmutableSet<String> TVOS_DEVICE_TARGET_CPUS =
      ImmutableSet.of("tvos_arm64");
  private static final ImmutableSet<String> CATALYST_TARGET_CPUS =
      ImmutableSet.of("catalyst_x86_64");
  private static final ImmutableSet<String> MACOS_TARGET_CPUS =
      ImmutableSet.of("darwin_x86_64", "darwin_arm64", "darwin_arm64e");

  private final String starlarkKey;
  private final String nameInPlist;
  private final String platformType;
  private final boolean isDevice;

  ApplePlatform(String starlarkKey, String nameInPlist, String platformType, boolean isDevice) {
    this.starlarkKey = starlarkKey;
    this.nameInPlist = Preconditions.checkNotNull(nameInPlist);
    this.platformType = platformType;
    this.isDevice = isDevice;
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  @Override
  public String getName() {
    return starlarkKey;
  }

  @Override
  public String getType() {
    return platformType;
  }

  @Override
  public boolean isDevice() {
    return isDevice;
  }

  @Override
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
  private static ApplePlatform forTargetCpuNullable(String targetCpu) {
    if (IOS_SIMULATOR_TARGET_CPUS.contains(targetCpu)) {
      return IOS_SIMULATOR;
    } else if (IOS_DEVICE_TARGET_CPUS.contains(targetCpu)) {
      return IOS_DEVICE;
    } else if (VISIONOS_SIMULATOR_TARGET_CPUS.contains(targetCpu)) {
      return VISIONOS_SIMULATOR;
    } else if (VISIONOS_DEVICE_TARGET_CPUS.contains(targetCpu)) {
      return VISIONOS_DEVICE;
    } else if (WATCHOS_SIMULATOR_TARGET_CPUS.contains(targetCpu)) {
      return WATCHOS_SIMULATOR;
    } else if (WATCHOS_DEVICE_TARGET_CPUS.contains(targetCpu)) {
      return WATCHOS_DEVICE;
    } else if (TVOS_SIMULATOR_TARGET_CPUS.contains(targetCpu)) {
      return TVOS_SIMULATOR;
    } else if (TVOS_DEVICE_TARGET_CPUS.contains(targetCpu)) {
      return TVOS_DEVICE;
    } else if (CATALYST_TARGET_CPUS.contains(targetCpu)) {
      return CATALYST;
    } else if (MACOS_TARGET_CPUS.contains(targetCpu)) {
      return MACOS;
    } else {
      return null;
    }
  }

  /**
   * Returns the platform cpu string for the given target cpu and platform type.
   *
   * @param platformType platform type that the given cpu value is implied for
   * @param arch architecture representation, such as 'arm64'
   */
  private static String cpuStringForTarget(String platformType, String arch) {
    switch (platformType) {
      case PlatformType.MACOS:
        return String.format("darwin_%s", arch);
      default:
        return String.format("%s_%s", platformType.toString(), arch);
    }
  }

  /**
   * Returns the platform for the given target cpu and platform type.
   *
   * @param platformType platform type that the given cpu value is implied for
   * @param arch architecture representation, such as 'arm64'
   * @throws IllegalArgumentException if there is no valid apple platform for the given target cpu
   */
  public static ApplePlatform forTarget(String platformType, String arch) {
    return forTargetCpu(cpuStringForTarget(platformType, arch));
  }

  /**
   * Returns the platform for the given target cpu.
   *
   * @param targetCpu cpu value with platform type prefix, such as 'ios_arm64'
   * @throws IllegalArgumentException if there is no valid apple platform for the given target cpu
   */
  public static ApplePlatform forTargetCpu(String targetCpu) {
    ApplePlatform platform = forTargetCpuNullable(targetCpu);
    if (platform != null) {
      return platform;
    } else {
      throw new IllegalArgumentException(
          "No supported apple platform registered for target cpu " + targetCpu);
    }
  }

  /** Returns a Starlark struct that contains the instances of this enum. */
  public static StructImpl getStarlarkStruct() {
    Provider constructor = new BuiltinProvider<StructImpl>("platforms", StructImpl.class) {};
    HashMap<String, Object> fields = new HashMap<>();
    for (ApplePlatform type : values()) {
      fields.put(type.starlarkKey, type);
    }
    return StarlarkInfo.create(constructor, fields, Location.BUILTIN);
  }

  @Override
  public void repr(Printer printer) {
    printer.append(Ascii.toLowerCase(toString()));
  }

  /** Exception indicating an unknown or unsupported Apple platform type. */
  public static class UnsupportedPlatformTypeException extends Exception {
    public UnsupportedPlatformTypeException(String msg) {
      super(msg);
    }
  }

  // TODO(b/331163027): Remove this duplicate of common/objc/apple_platform.PLATFORM_TYPE
  /**
   * The former enum PlatformType is being migrated to a Starlark struct PLATFORM_TYPE in
   * builtins_bzl/common/objc/apple_platform.bzl. During the migration, PlatformType has been
   * converted to a static class hosting string constants as Java duplicates of
   * apple_platform.PLATFORM_TYPE.
   */
  public static class PlatformType { // implements ApplePlatformTypeApi {
    public static final String IOS = "ios";
    public static final String VISIONOS = "visionos";
    public static final String WATCHOS = "watchos";
    public static final String TVOS = "tvos";
    public static final String MACOS = "macos";
    public static final String CATALYST = "catalyst";

    private PlatformType() {}
  }
  // LINT.ThenChange(//src/main/starlark/builtins_bzl/common/objc/apple_platform.bzl)
}
