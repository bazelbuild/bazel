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

package com.google.devtools.build.lib.rules.platform;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link ConstraintSetting} and {@link ConstraintValue}. */
@RunWith(JUnit4.class)
public class ConstraintTest extends BuildViewTestCase {

  @Before
  public void createConstraints() throws Exception {
    scratch.file(
        "constraint/BUILD",
        "constraint_setting(name = 'basic')",
        "constraint_value(name = 'foo',",
        "    constraint_setting = ':basic',",
        "    )",
        "constraint_value(name = 'bar',",
        "    constraint_setting = ':basic',",
        "    )");
  }

  @Test
  public void testConstraint() throws Exception {
    ConfiguredTarget setting = getConfiguredTarget("//constraint:basic");
    assertThat(setting).isNotNull();

    ConstraintSettingInfo constraintSettingInfo = PlatformProviderUtils.constraintSetting(setting);
    assertThat(constraintSettingInfo).isNotNull();
    assertThat(constraintSettingInfo).isNotNull();
    assertThat(constraintSettingInfo.label())
        .isEqualTo(Label.parseAbsolute("//constraint:basic", ImmutableMap.of()));
    assertThat(constraintSettingInfo.defaultConstraintValue()).isNull();

    ConfiguredTarget fooValue = getConfiguredTarget("//constraint:foo");
    assertThat(fooValue).isNotNull();

    ConstraintValueInfo fooConstraintValueInfo = PlatformProviderUtils.constraintValue(fooValue);
    assertThat(fooConstraintValueInfo).isNotNull();
    assertThat(fooConstraintValueInfo.constraint().label())
        .isEqualTo(Label.parseAbsolute("//constraint:basic", ImmutableMap.of()));
    assertThat(fooConstraintValueInfo.label())
        .isEqualTo(Label.parseAbsolute("//constraint:foo", ImmutableMap.of()));

    ConfiguredTarget barValue = getConfiguredTarget("//constraint:bar");
    assertThat(barValue).isNotNull();

    ConstraintValueInfo barConstraintValueInfo = PlatformProviderUtils.constraintValue(barValue);
    assertThat(barConstraintValueInfo.constraint().label())
        .isEqualTo(Label.parseAbsolute("//constraint:basic", ImmutableMap.of()));
    assertThat(barConstraintValueInfo.label())
        .isEqualTo(Label.parseAbsolute("//constraint:bar", ImmutableMap.of()));
  }

  @Test
  public void testConstraint_defaultValue() throws Exception {
    scratch.file(
        "constraint_default/BUILD",
        "constraint_setting(name = 'basic',",
        "    default_constraint_value = ':foo',",
        "    )",
        "constraint_value(name = 'foo',",
        "    constraint_setting = ':basic',",
        "    )",
        "constraint_value(name = 'bar',",
        "    constraint_setting = ':basic',",
        "    )");

    ConfiguredTarget setting = getConfiguredTarget("//constraint_default:basic");
    assertThat(setting).isNotNull();
    ConstraintSettingInfo constraintSettingInfo = PlatformProviderUtils.constraintSetting(setting);
    assertThat(constraintSettingInfo).isNotNull();

    ConfiguredTarget fooValue = getConfiguredTarget("//constraint_default:foo");
    assertThat(fooValue).isNotNull();
    ConstraintValueInfo fooConstraintValueInfo = PlatformProviderUtils.constraintValue(fooValue);
    assertThat(fooConstraintValueInfo).isNotNull();

    assertThat(constraintSettingInfo.defaultConstraintValue()).isEqualTo(fooConstraintValueInfo);
  }
}
