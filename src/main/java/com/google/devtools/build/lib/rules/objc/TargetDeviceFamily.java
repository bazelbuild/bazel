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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;

import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Possible values in the {@code TARGETED_DEVICE_FAMILY} build setting.
 */
public enum TargetDeviceFamily {
  IPAD, IPHONE, WATCH;

  /**
   * An exception that indicates the name of a device family was not recognized or is somehow
   * invalid.
   */
  public static class InvalidFamilyNameException extends IllegalArgumentException {
    public InvalidFamilyNameException(String message) {
      super(message);
    }
  }

  /**
   * An exception that indicates a family name appeared twice in a sequence when only one is
   * expected.
   */
  public static class RepeatedFamilyNameException extends IllegalArgumentException {
    public RepeatedFamilyNameException(String message) {
      super(message);
    }
  }

  /**
   * Contains the values of the UIDeviceFamily plist info setting for each valid set of
   * TargetDeviceFamilies.
   */
  public static final Map<Set<TargetDeviceFamily>, List<Integer>> UI_DEVICE_FAMILY_VALUES =
      ImmutableMap.<Set<TargetDeviceFamily>, List<Integer>>builder()
          .put(ImmutableSet.of(TargetDeviceFamily.IPHONE), ImmutableList.of(1))
          .put(ImmutableSet.of(TargetDeviceFamily.IPAD), ImmutableList.of(2))
          .put(ImmutableSet.of(TargetDeviceFamily.WATCH), ImmutableList.of(4))
          .put(ImmutableSet.of(TargetDeviceFamily.IPHONE, TargetDeviceFamily.IPAD),
              ImmutableList.of(1, 2))
          .put(ImmutableSet.of(TargetDeviceFamily.IPHONE, TargetDeviceFamily.WATCH),
              ImmutableList.of(1, 4))
          .build();

  /**
   * Returns the name of the family as it appears in build rules.
   */
  public String getNameInRule() {
    return BY_NAME_IN_RULE.get(this);
  }

  private static final ImmutableBiMap<TargetDeviceFamily, String> BY_NAME_IN_RULE =
      ImmutableBiMap.of(IPAD, "ipad", IPHONE, "iphone", WATCH, "watch");

  /**
   * Converts a sequence containing the strings returned by {@link #getNameInRule()} to a set of
   * instances of this enum.
   *
   * <p>If there are multiple items in the returned set, they are in enumeration order.
   *
   * @param names the names of the families
   * @throws InvalidFamilyNameException if some family name in the sequence was not recognized
   * @throws RepeatedFamilyNameException if some family name appeared in the sequence twice
   */
  public static Set<TargetDeviceFamily> fromNamesInRule(Iterable<String> names) {
    Set<TargetDeviceFamily> families = EnumSet.noneOf(TargetDeviceFamily.class);
    for (String name : names) {
      TargetDeviceFamily family = BY_NAME_IN_RULE.inverse().get(name);
      if (family == null) {
        throw new InvalidFamilyNameException(name);
      }
      if (!families.add(family)) {
        throw new RepeatedFamilyNameException(name);
      }
    }
    return families;
  }

  /**
   * Converts the {@code TARGETED_DEVICE_FAMILY} setting in build settings to a set of
   * {@code TargetedDevice}s.
   */
  public static Set<TargetDeviceFamily> fromBuildSetting(String targetedDevice) {
    ImmutableSet.Builder<TargetDeviceFamily> result = ImmutableSet.builder();
    for (String numericSetting : Splitter.on(',').split(targetedDevice)) {
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
