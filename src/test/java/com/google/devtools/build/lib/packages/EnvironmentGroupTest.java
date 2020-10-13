// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link EnvironmentGroup}. Note input validation is handled in
 * {@link PackageFactoryTest}.
 */
@RunWith(JUnit4.class)
public class EnvironmentGroupTest extends PackageLoadingTestCase {

  private EnvironmentGroup group;

  @Before
  public final void createPackage() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "environment(name='foo', fulfills = [':bar', ':baz'])",
        "environment(name='bar', fulfills = [':baz'])",
        "environment(name='baz')",
        "environment(name='not_in_group')",
        "environment_group(",
        "    name = 'group',",
        "    environments = [':foo', ':bar', ':baz'],",
        "    defaults = [':foo'],",
        ")");
    group = (EnvironmentGroup) getTarget("//pkg:group");
  }

  @Test
  public void testGroupMembership() throws Exception {
    assertThat(group.getEnvironments())
        .isEqualTo(
            ImmutableSet.of(
                Label.parseAbsolute("//pkg:foo", ImmutableMap.of()),
                Label.parseAbsolute("//pkg:bar", ImmutableMap.of()),
                Label.parseAbsolute("//pkg:baz", ImmutableMap.of())));
  }

  @Test
  public void defaultsMembership() throws Exception {
    assertThat(group.getDefaults())
        .isEqualTo(ImmutableSet.of(Label.parseAbsolute("//pkg:foo", ImmutableMap.of())));
  }

  @Test
  public void isDefault() throws Exception {
    EnvironmentLabels unpackedGroup = group.getEnvironmentLabels();
    assertThat(unpackedGroup.isDefault(Label.parseAbsolute("//pkg:foo", ImmutableMap.of())))
        .isTrue();
    assertThat(unpackedGroup.isDefault(Label.parseAbsolute("//pkg:bar", ImmutableMap.of())))
        .isFalse();
    assertThat(unpackedGroup.isDefault(Label.parseAbsolute("//pkg:baz", ImmutableMap.of())))
        .isFalse();
    assertThat(
            unpackedGroup.isDefault(Label.parseAbsolute("//pkg:not_in_group", ImmutableMap.of())))
        .isFalse();
  }

  @Test
  public void fulfillers() throws Exception {
    EnvironmentLabels unpackedGroup = group.getEnvironmentLabels();
    assertThat(
            unpackedGroup
                .getFulfillers(Label.parseAbsolute("//pkg:baz", ImmutableMap.of()))
                .toList())
        .containsExactly(
            Label.parseAbsolute("//pkg:foo", ImmutableMap.of()),
            Label.parseAbsolute("//pkg:bar", ImmutableMap.of()));
    assertThat(
            unpackedGroup
                .getFulfillers(Label.parseAbsolute("//pkg:bar", ImmutableMap.of()))
                .toList())
        .containsExactly(Label.parseAbsolute("//pkg:foo", ImmutableMap.of()));
    assertThat(
            unpackedGroup
                .getFulfillers(Label.parseAbsolute("//pkg:foo", ImmutableMap.of()))
                .toList())
        .isEmpty();
  }

  @Test
  public void emptyGroupsNotAllowed() throws Exception {
    scratch.file(
        "a/BUILD", "environment_group(name = 'empty_group', environments = [], defaults = [])");
    reporter.removeHandler(failFastHandler);
    Package pkg = getTarget("//a:BUILD").getPackage();
    assertThat(pkg.containsErrors()).isTrue();
    assertContainsEvent(
        "environment group empty_group must contain at least one environment");
  }
}
