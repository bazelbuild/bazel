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
import com.google.devtools.build.lib.rules.platform.ConstraintValue;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link ConstraintValueInfo}. */
@RunWith(JUnit4.class)
public class ConstraintValueInfoTest extends BuildViewTestCase {

  @Test
  public void constraintValue_equalsTester() {
    ConstraintSettingInfo setting1 = ConstraintSettingInfo.create(makeLabel("//constraint:basic"));
    ConstraintSettingInfo setting2 = ConstraintSettingInfo.create(makeLabel("//constraint:other"));
    new EqualsTester()
        .addEqualityGroup(
            // Base case.
            ConstraintValueInfo.create(setting1, makeLabel("//constraint:value")),
            ConstraintValueInfo.create(setting1, makeLabel("//constraint:value")))
        .addEqualityGroup(
            // Different label.
            ConstraintValueInfo.create(setting1, makeLabel("//constraint:otherValue")))
        .addEqualityGroup(
            // Different setting.
            ConstraintValueInfo.create(setting2, makeLabel("//constraint:otherValue")))
        .testEquals();
  }

  @Test
  public void constraintValueInfoConstructor() throws Exception {
    scratch.file(
        "test/platform/my_constraint_value.bzl",
        "def _impl(ctx):",
        "  setting = ctx.attr.setting[platform_common.ConstraintSettingInfo]",
        "  constraint_value = platform_common.ConstraintValueInfo(",
        "    label = ctx.label, constraint_setting = setting)",
        "  return [constraint_value]",
        "my_constraint_value = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'setting': attr.label(providers = [platform_common.ConstraintSettingInfo]),",
        "  }",
        ")");
    scratch.file(
        "test/platform/BUILD",
        "load('//test/platform:my_constraint_value.bzl', 'my_constraint_value')",
        "constraint_setting(name = 'basic')",
        "my_constraint_value(",
        "  name = 'custom',",
        "  setting = ':basic',",
        ")");

    ConfiguredTarget value = getConfiguredTarget("//test/platform:custom");
    assertThat(value).isNotNull();
    assertThat(ConstraintValue.constraintValue(value)).isNotNull();
    assertThat(ConstraintValue.constraintValue(value).constraint().label())
        .isEqualTo(Label.parseAbsolute("//test/platform:basic"));
    assertThat(ConstraintValue.constraintValue(value).label())
        .isEqualTo(Label.parseAbsolute("//test/platform:custom"));
  }
}
