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

package com.google.devtools.build.lib.analysis.platform;


import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link ConstraintValueInfo}. */
@RunWith(JUnit4.class)
public class ConstraintValueInfoTest extends BuildViewTestCase {

  @Test
  public void constraintValue_equalsTester() {
    ConstraintSettingInfo setting1 =
        ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("//constraint:basic"));
    ConstraintSettingInfo setting2 =
        ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("//constraint:other"));
    new EqualsTester()
        .addEqualityGroup(
            // Base case.
            ConstraintValueInfo.create(
                setting1, Label.parseAbsoluteUnchecked("//constraint:value")),
            ConstraintValueInfo.create(
                setting1, Label.parseAbsoluteUnchecked("//constraint:value")))
        .addEqualityGroup(
            // Different label.
            ConstraintValueInfo.create(
                setting1, Label.parseAbsoluteUnchecked("//constraint:otherValue")))
        .addEqualityGroup(
            // Different setting.
            ConstraintValueInfo.create(
                setting2, Label.parseAbsoluteUnchecked("//constraint:otherValue")))
        .testEquals();
  }
}
