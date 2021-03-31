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

/** Tests of {@link ConstraintSettingInfo}. */
@RunWith(JUnit4.class)
public class ConstraintSettingInfoTest extends BuildViewTestCase {

  @Test
  public void constraintSetting_equalsTester() {
    new EqualsTester()
        .addEqualityGroup(
            ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("//constraint:basic")),
            ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("//constraint:basic")))
        .addEqualityGroup(
            ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("//constraint:other")))
        .testEquals();
  }
}
