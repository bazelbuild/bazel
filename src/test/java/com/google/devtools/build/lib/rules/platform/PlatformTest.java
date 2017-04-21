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
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.platform.PlatformInfo.DuplicateConstraintException;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link Platform}. */
@RunWith(JUnit4.class)
public class PlatformTest extends BuildViewTestCase {

  @Rule public ExpectedException expectedException = ExpectedException.none();

  @Before
  public void createPlatform() throws Exception {
    scratch.file(
        "constraint/BUILD",
        "constraint_setting(name = 'basic')",
        "constraint_value(name = 'foo',",
        "    constraint_setting = ':basic',",
        "    )",
        "platform(name = 'plat1',",
        "    constraint_values = [",
        "       ':foo',",
        "    ])");
  }

  @Test
  public void testPlatform() throws Exception {
    ConfiguredTarget platform = getConfiguredTarget("//constraint:plat1");
    assertThat(platform).isNotNull();

    PlatformInfo provider = PlatformInfo.fromTarget(platform);
    assertThat(provider).isNotNull();
    assertThat(provider.constraints()).hasSize(1);
    ConstraintSettingInfo constraintSetting =
        ConstraintSettingInfo.create(makeLabel("//constraint:basic"));
    ConstraintValueInfo constraintValue =
        ConstraintValueInfo.create(constraintSetting, makeLabel("//constraint:foo"));
    assertThat(provider.constraints()).containsExactly(constraintValue);
    assertThat(provider.remoteExecutionProperties()).isEmpty();
  }

  @Test
  public void testPlatform_overlappingConstraintValueError() throws Exception {
    checkError(
        "constraint/overlap",
        "plat_overlap",
        "Duplicate constraint_values for constraint_setting //constraint:basic: "
            + "//constraint:foo, //constraint/overlap:bar",
        "constraint_value(name = 'bar',",
        "    constraint_setting = '//constraint:basic',",
        "    )",
        "platform(name = 'plat_overlap',",
        "    constraint_values = [",
        "       '//constraint:foo',",
        "       ':bar',",
        "    ])");
  }

  @Test
  public void testPlatform_remoteExecution() throws Exception {
    scratch.file(
        "constraint/remote/BUILD",
        "platform(name = 'plat_remote',",
        "    constraint_values = [",
        "       '//constraint:foo',",
        "    ],",
        "    remote_execution_properties = {",
        "        'foo': 'val1',",
        "        'bar': 'val2',",
        "    },",
        ")");

    ConfiguredTarget platform = getConfiguredTarget("//constraint/remote:plat_remote");
    assertThat(platform).isNotNull();

    PlatformInfo provider = PlatformInfo.fromTarget(platform);
    assertThat(provider).isNotNull();
    assertThat(provider.remoteExecutionProperties())
        .containsExactlyEntriesIn(ImmutableMap.of("foo", "val1", "bar", "val2"));
  }

  @Test
  public void testPlatform_skylark() throws Exception {

    scratch.file(
        "test/platform/platform.bzl",
        "def _impl(ctx):",
        "  platform = ctx.attr.platform[platform_common.PlatformInfo]",
        "  return struct(",
        "    count = len(platform.constraints),",
        "    first_setting = platform.constraints[0].constraint.label,",
        "    first_value = platform.constraints[0].label)",
        "my_rule = rule(",
        "  _impl,",
        "  attrs = { 'platform': attr.label(providers = [platform_common.PlatformInfo])},",
        ")");

    scratch.file(
        "test/platform/BUILD",
        "load('//test/platform:platform.bzl', 'my_rule')",
        "my_rule(name = 'r',",
        "  platform = '//constraint:plat1')");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//test/platform:r");
    assertThat(configuredTarget).isNotNull();

    int count = (int) configuredTarget.get("count");
    assertThat(count).isEqualTo(1);

    Label settingLabel = (Label) configuredTarget.get("first_setting");
    assertThat(settingLabel).isNotNull();
    assertThat(settingLabel).isEqualTo(makeLabel("//constraint:basic"));
    Label valueLabel = (Label) configuredTarget.get("first_value");
    assertThat(valueLabel).isNotNull();
    assertThat(valueLabel).isEqualTo(makeLabel("//constraint:foo"));
  }

  @Test
  public void platformInfo_overlappingConstraintsError() throws DuplicateConstraintException {
    ConstraintSettingInfo setting = ConstraintSettingInfo.create(makeLabel("//constraint:basic"));

    ConstraintValueInfo value1 = ConstraintValueInfo.create(setting, makeLabel("//constraint:foo"));
    ConstraintValueInfo value2 = ConstraintValueInfo.create(setting, makeLabel("//constraint:bar"));

    PlatformInfo.Builder builder =
        PlatformInfo.builder().addConstraint(value1).addConstraint(value2);

    expectedException.expect(DuplicateConstraintException.class);
    expectedException.expectMessage(
        "Duplicate constraint_values for constraint_setting //constraint:basic: "
            + "//constraint:foo, //constraint:bar");
    builder.build();
  }

  @Test
  public void platformInfo_equalsTester() throws DuplicateConstraintException {
    ConstraintSettingInfo setting1 = ConstraintSettingInfo.create(makeLabel("//constraint:basic"));
    ConstraintSettingInfo setting2 = ConstraintSettingInfo.create(makeLabel("//constraint:other"));

    ConstraintValueInfo value1 =
        ConstraintValueInfo.create(setting1, makeLabel("//constraint:value1"));
    ConstraintValueInfo value2 =
        ConstraintValueInfo.create(setting2, makeLabel("//constraint:value2"));
    ConstraintValueInfo value3 =
        ConstraintValueInfo.create(setting2, makeLabel("//constraint:value3"));

    new EqualsTester()
        .addEqualityGroup(
            // Base case.
            PlatformInfo.builder().addConstraint(value1).addConstraint(value2).build(),
            PlatformInfo.builder().addConstraint(value1).addConstraint(value2).build(),
            PlatformInfo.builder()
                .addConstraint(value1)
                .addConstraint(value2)
                .addRemoteExecutionProperty("key", "val") // execution properties are ignored.
                .build())
        .addEqualityGroup(
            // Extra constraint.
            PlatformInfo.builder().addConstraint(value1).addConstraint(value3).build())
        .addEqualityGroup(
            // Missing constraint.
            PlatformInfo.builder().addConstraint(value1).build())
        .testEquals();
  }
}
