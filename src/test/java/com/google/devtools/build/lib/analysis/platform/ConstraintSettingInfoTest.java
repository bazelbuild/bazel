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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
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
            ConstraintSettingInfo.create(makeLabel("//constraint:basic")),
            ConstraintSettingInfo.create(makeLabel("//constraint:basic")))
        .addEqualityGroup(ConstraintSettingInfo.create(makeLabel("//constraint:other")))
        .testEquals();
  }

  @Test
  public void constraintSettingInfoConstructor() throws Exception {
    scratch.file(
        "test/platform/my_constraint_setting.bzl",
        "def _impl(ctx):",
        "  constraint_setting = platform_common.ConstraintSettingInfo(label = ctx.label)",
        "  return [constraint_setting]",
        "my_constraint_setting = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "  }",
        ")");
    scratch.file(
        "test/platform/BUILD",
        "load('//test/platform:my_constraint_setting.bzl', 'my_constraint_setting')",
        "my_constraint_setting(name = 'custom')");

    ConfiguredTarget setting = getConfiguredTarget("//test/platform:custom");
    assertThat(setting).isNotNull();
    assertThat(PlatformProviderUtils.constraintSetting(setting)).isNotNull();
    assertThat(PlatformProviderUtils.constraintSetting(setting).label())
        .isEqualTo(Label.parseAbsolute("//test/platform:custom"));
  }
}
