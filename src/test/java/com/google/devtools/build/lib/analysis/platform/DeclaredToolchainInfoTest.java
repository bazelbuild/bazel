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
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link DeclaredToolchainInfo}. */
@RunWith(JUnit4.class)
public class DeclaredToolchainInfoTest extends BuildViewTestCase {

  @Test
  public void toolchainInfo_equalsTester() throws Exception {
    ConstraintSettingInfo setting1 =
        ConstraintSettingInfo.create(makeLabel("//constraint:setting1"));
    ConstraintValueInfo constraint1 =
        ConstraintValueInfo.create(setting1, makeLabel("//constraint:foo"));
    ConstraintValueInfo constraint2 =
        ConstraintValueInfo.create(setting1, makeLabel("//constraint:bar"));

    new EqualsTester()
        .addEqualityGroup(
            // Base case.
            DeclaredToolchainInfo.create(
                ToolchainTypeInfo.create(makeLabel("//toolchain:tc1")),
                ImmutableList.of(constraint1),
                ImmutableList.of(constraint2),
                makeLabel("//toolchain:toolchain1")),
            DeclaredToolchainInfo.create(
                ToolchainTypeInfo.create(makeLabel("//toolchain:tc1")),
                ImmutableList.of(constraint1),
                ImmutableList.of(constraint2),
                makeLabel("//toolchain:toolchain1")))
        .addEqualityGroup(
            // Different type.
            DeclaredToolchainInfo.create(
                ToolchainTypeInfo.create(makeLabel("//toolchain:tc2")),
                ImmutableList.of(constraint1),
                ImmutableList.of(constraint2),
                makeLabel("//toolchain:toolchain1")))
        .addEqualityGroup(
            // Different constraints.
            DeclaredToolchainInfo.create(
                ToolchainTypeInfo.create(makeLabel("//toolchain:tc1")),
                ImmutableList.of(constraint2),
                ImmutableList.of(constraint1),
                makeLabel("//toolchain:toolchain1")))
        .addEqualityGroup(
            // Different toolchain label.
            DeclaredToolchainInfo.create(
                ToolchainTypeInfo.create(makeLabel("//toolchain:tc1")),
                ImmutableList.of(constraint1),
                ImmutableList.of(constraint2),
                makeLabel("//toolchain:toolchain2")))
        .testEquals();
  }
}
