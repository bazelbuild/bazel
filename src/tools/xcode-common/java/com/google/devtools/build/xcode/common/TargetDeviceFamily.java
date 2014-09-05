// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.xcode.common;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Possible values in the {@code TARGETED_DEVICE_FAMILY} build setting.
 */
public enum TargetDeviceFamily {
  IPAD("ipad"), IPHONE("iphone");

  /**
   * Contains the values of the UIDeviceFamily plist info setting for each valid set of
   * TargetDeviceFamilies.
   */
  public static final Map<Set<TargetDeviceFamily>, List<Integer>> UI_DEVICE_FAMILY_VALUES =
      ImmutableMap.<Set<TargetDeviceFamily>, List<Integer>>builder()
          .put(ImmutableSet.of(TargetDeviceFamily.IPHONE), ImmutableList.of(1))
          .put(ImmutableSet.of(TargetDeviceFamily.IPAD), ImmutableList.of(2))
          .put(ImmutableSet.of(TargetDeviceFamily.IPHONE, TargetDeviceFamily.IPAD),
              ImmutableList.of(1, 2))
          .build();

  private final String nameInRule;

  TargetDeviceFamily(String nameInRule) {
    this.nameInRule = Preconditions.checkNotNull(nameInRule);
  }

  public String getNameInRule() {
    return nameInRule;
  }

  /**
   * Converts the {@code TARGETED_DEVICE_FAMILY} setting in build settings to a set of
   * {@code TargetedDevice}s.
   */
  public static Set<TargetDeviceFamily> fromBuildSetting(String targetedDevice) {
    ImmutableSet.Builder<TargetDeviceFamily> result = ImmutableSet.builder();
    for (String numericSetting : Splitter.on(",").split(targetedDevice)) {
      numericSetting = numericSetting.trim();
      switch (numericSetting) {
        case "1":
          result.add(IPHONE);
          break;
        case "2":
          result.add(IPAD);
          break;
        default:
          throw new IllegalArgumentException(
              "Expect comma-separated list containing only '1' and/or '2' for "
              + "TARGETED_DEVICE_FAMILY: " + targetedDevice);
      }
    }
    return result.build();
  }
}
