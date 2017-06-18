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
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link ToolchainInfo}. */
@RunWith(JUnit4.class)
public class ToolchainInfoTest extends BuildViewTestCase {

  @Test
  public void toolchainInfo_equalsTester() throws Exception {
    ClassObjectConstructor.Key key = new ClassObjectConstructor.Key() {};
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
                key,
                ImmutableList.of(value1, value2),
                ImmutableList.of(value1, value3),
                ImmutableMap.<String, Object>of("foo", "val1", "bar", "val2"),
                Location.BUILTIN),
            new ToolchainInfo(
                key,
                ImmutableList.of(value1, value2),
                ImmutableList.of(value1, value3),
                ImmutableMap.<String, Object>of("foo", "val1", "bar", "val2"),
                Location.BUILTIN))
        .addEqualityGroup(
            // Different type.
            new ToolchainInfo(
                new ClassObjectConstructor.Key() {},
                ImmutableList.of(value1, value2),
                ImmutableList.of(value1, value3),
                ImmutableMap.<String, Object>of("foo", "val1", "bar", "val2"),
                Location.BUILTIN))
        .addEqualityGroup(
            // Different target constraints.
            new ToolchainInfo(
                key,
                ImmutableList.of(value1, value2),
                ImmutableList.of(value1, value2),
                ImmutableMap.<String, Object>of("foo", "val1", "bar", "val2"),
                Location.BUILTIN))
        .addEqualityGroup(
            // Different data.
            new ToolchainInfo(
                key,
                ImmutableList.of(value1, value2),
                ImmutableList.of(value1, value3),
                ImmutableMap.<String, Object>of("foo", "val1", "bar", "val3"),
                Location.BUILTIN))
        .testEquals();
  }
}
