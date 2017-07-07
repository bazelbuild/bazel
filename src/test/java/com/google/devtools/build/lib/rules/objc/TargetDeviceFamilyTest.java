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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.rules.objc.TargetDeviceFamily.InvalidFamilyNameException;
import com.google.devtools.build.lib.rules.objc.TargetDeviceFamily.RepeatedFamilyNameException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link TargetDeviceFamily}.
 */
@RunWith(JUnit4.class)
public class TargetDeviceFamilyTest {
  @Test
  public void uiDeviceFamilyValuesUndefinedForEmpty() {
    assertThat(TargetDeviceFamily.UI_DEVICE_FAMILY_VALUES.keySet())
        .doesNotContain(ImmutableSet.<TargetDeviceFamily>of());
  }

  @Test
  public void fromBuildSettings() {
    assertThat(TargetDeviceFamily.fromBuildSetting("1,2"))
        .isEqualTo(ImmutableSet.of(TargetDeviceFamily.IPAD, TargetDeviceFamily.IPHONE));
    assertThat(TargetDeviceFamily.fromBuildSetting(" 1, 2"))
        .isEqualTo(ImmutableSet.of(TargetDeviceFamily.IPAD, TargetDeviceFamily.IPHONE));
    assertThat(TargetDeviceFamily.fromBuildSetting("1,2\n"))
        .isEqualTo(ImmutableSet.of(TargetDeviceFamily.IPAD, TargetDeviceFamily.IPHONE));
    assertThat(TargetDeviceFamily.fromBuildSetting("1"))
        .isEqualTo(ImmutableSet.of(TargetDeviceFamily.IPHONE));
    assertThat(TargetDeviceFamily.fromBuildSetting("2"))
        .isEqualTo(ImmutableSet.of(TargetDeviceFamily.IPAD));
  }

  private void checkFromNamesInRuleThrows(
      Class<? extends Exception> expectedClass, String... names) {
    try {
      TargetDeviceFamily.fromNamesInRule(ImmutableList.copyOf(names));
      fail("should have thrown");
    } catch (IllegalArgumentException expected) {
      assertThat(expected.getClass()).isEqualTo(expectedClass);
    }
  }

  @Test
  public void fromNamesInRule_errors() {
    checkFromNamesInRuleThrows(InvalidFamilyNameException.class, "foo");
    checkFromNamesInRuleThrows(InvalidFamilyNameException.class, "foo", "bar");
    checkFromNamesInRuleThrows(InvalidFamilyNameException.class, "iphone", "ipad", "bar");
    checkFromNamesInRuleThrows(RepeatedFamilyNameException.class, "iphone", "iphone");
    checkFromNamesInRuleThrows(RepeatedFamilyNameException.class, "ipad", "ipad");
  }

  @Test
  public void fromNamesInRule() {
    assertThat(TargetDeviceFamily.fromNamesInRule(ImmutableList.<String>of()))
        .isEmpty();
    assertThat(TargetDeviceFamily.fromNamesInRule(ImmutableList.of("iphone", "ipad")))
        .containsExactly(TargetDeviceFamily.IPAD, TargetDeviceFamily.IPHONE)
        .inOrder();
    assertThat(TargetDeviceFamily.fromNamesInRule(ImmutableList.of("ipad", "iphone")))
        .containsExactly(TargetDeviceFamily.IPAD, TargetDeviceFamily.IPHONE)
        .inOrder();
    assertThat(TargetDeviceFamily.fromNamesInRule(ImmutableList.of("iphone")))
        .containsExactly(TargetDeviceFamily.IPHONE);
    assertThat(TargetDeviceFamily.fromNamesInRule(ImmutableList.of("ipad")))
        .containsExactly(TargetDeviceFamily.IPAD);
  }
}
