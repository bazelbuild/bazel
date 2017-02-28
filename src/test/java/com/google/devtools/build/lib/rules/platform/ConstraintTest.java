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

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link ConstraintSetting} and {@link ConstraintValue}. */
@RunWith(JUnit4.class)
public class ConstraintTest extends BuildViewTestCase {

  @Test
  public void testConstraint() throws Exception {
    scratch.file(
        "constraint/BUILD",
        "constraint_setting(name = 'basic')",
        "constraint_value(name = 'foo',",
        "    constraint_setting = ':basic',",
        "    )",
        "constraint_value(name = 'bar',",
        "    constraint_setting = ':basic',",
        "    )");
    ConfiguredTarget setting = getConfiguredTarget("//constraint:basic");
    assertThat(setting).isNotNull();
    assertThat(setting.getProvider(ConstraintSettingProvider.class)).isNotNull();
    assertThat(setting.getProvider(ConstraintSettingProvider.class).constraintSetting())
        .isEqualTo(Label.parseAbsolute("//constraint:basic"));
    ConfiguredTarget fooValue = getConfiguredTarget("//constraint:foo");
    assertThat(fooValue).isNotNull();
    assertThat(fooValue.getProvider(ConstraintValueProvider.class)).isNotNull();
    assertThat(fooValue.getProvider(ConstraintValueProvider.class).constraint().constraintSetting())
        .isEqualTo(Label.parseAbsolute("//constraint:basic"));
    assertThat(fooValue.getProvider(ConstraintValueProvider.class).value())
        .isEqualTo(Label.parseAbsolute("//constraint:foo"));
    ConfiguredTarget barValue = getConfiguredTarget("//constraint:bar");
    assertThat(barValue).isNotNull();
    assertThat(barValue.getProvider(ConstraintValueProvider.class).constraint().constraintSetting())
        .isEqualTo(Label.parseAbsolute("//constraint:basic"));
    assertThat(barValue.getProvider(ConstraintValueProvider.class).value())
        .isEqualTo(Label.parseAbsolute("//constraint:bar"));
  }
}
