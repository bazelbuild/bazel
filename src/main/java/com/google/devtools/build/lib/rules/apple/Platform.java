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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;

import java.util.Locale;
import java.util.Set;

/**
 * An enum that can be used to distinguish between various apple platforms.
 */
public enum Platform {
  IOS_DEVICE("iPhoneOS"), IOS_SIMULATOR("iPhoneSimulator");

  private static final Set<String> IOS_SIMULATOR_ARCHS = ImmutableSet.of("i386", "x86_64");
  private static final Set<String> IOS_DEVICE_ARCHS =
      ImmutableSet.of("armv6", "armv7", "armv7s", "arm64");

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
   * Returns the platform for the architecture.
   * 
   * @throws IllegalArgumentException if there is no valid apple platform for the given
   *     architecture
   */
  public static Platform forArch(String arch) {
    if (IOS_SIMULATOR_ARCHS.contains(arch)) {
      return IOS_SIMULATOR;
    } else if (IOS_DEVICE_ARCHS.contains(arch)) {
      return IOS_DEVICE;
    } else {
      throw new IllegalArgumentException(
          "No supported apple platform registered for architecture " + arch);
    }
  }
}
