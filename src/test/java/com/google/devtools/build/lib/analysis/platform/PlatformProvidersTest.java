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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.events.Location;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of platform providers. */
@RunWith(JUnit4.class)
public class PlatformProvidersTest extends BuildViewTestCase {
  @Rule public ExpectedException expectedException = ExpectedException.none();

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
            ConstraintValueInfo.create(setting2, makeLabel("//constraint:ovalue")))
        .testEquals();
  }

  @Test
  public void platformInfo_overlappingConstraintsError() throws Exception {
    ConstraintSettingInfo setting = ConstraintSettingInfo.create(makeLabel("//constraint:basic"));

    ConstraintValueInfo value1 = ConstraintValueInfo.create(setting, makeLabel("//constraint:foo"));
    ConstraintValueInfo value2 = ConstraintValueInfo.create(setting, makeLabel("//constraint:bar"));

    PlatformInfo.Builder builder =
        PlatformInfo.builder().addConstraint(value1).addConstraint(value2);

    expectedException.expect(PlatformInfo.DuplicateConstraintException.class);
    expectedException.expectMessage(
        "Duplicate constraint_values for constraint_setting //constraint:basic: "
            + "//constraint:foo, //constraint:bar");
    builder.build();
  }

  @Test
  public void platformInfo_equalsTester() throws Exception {
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

  @Test
  public void toolchainInfo_equalsTester() throws Exception {
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
            new ToolchainInfo(
                ImmutableList.of(value1, value2),
                ImmutableList.of(value1, value3),
                ImmutableMap.<String, Object>of("foo", "val1", "bar", "val2"),
                Location.BUILTIN),
            new ToolchainInfo(
                ImmutableList.of(value1, value2),
                ImmutableList.of(value1, value3),
                ImmutableMap.<String, Object>of("foo", "val1", "bar", "val2"),
                Location.BUILTIN))
        .addEqualityGroup(
            // Different target constraints.
            new ToolchainInfo(
                ImmutableList.of(value1, value2),
                ImmutableList.of(value1, value2),
                ImmutableMap.<String, Object>of("foo", "val1", "bar", "val2"),
                Location.BUILTIN))
        .addEqualityGroup(
            // Different data.
            new ToolchainInfo(
                ImmutableList.of(value1, value2),
                ImmutableList.of(value1, value3),
                ImmutableMap.<String, Object>of("foo", "val1", "bar", "val3"),
                Location.BUILTIN))
        .testEquals();
  }
}
