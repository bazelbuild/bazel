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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.vfs.Path;

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

  private Package pkg;
  private EnvironmentGroup group;

  @Before
  public final void createPackage() throws Exception {
    Path buildfile =
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
    pkg =
        packageFactory.createPackageForTesting(
            PackageIdentifier.createInMainRepo("pkg"), buildfile, getPackageManager(), reporter);

    group = (EnvironmentGroup) pkg.getTarget("group");
  }

  @Test
  public void testGroupMembership() throws Exception {
    assertEquals(
        ImmutableSet.of(
            Label.parseAbsolute("//pkg:foo"),
            Label.parseAbsolute("//pkg:bar"),
            Label.parseAbsolute("//pkg:baz")),
        group.getEnvironments());
  }

  @Test
  public void testDefaultsMembership() throws Exception {
    assertEquals(ImmutableSet.of(Label.parseAbsolute("//pkg:foo")), group.getDefaults());
  }

  @Test
  public void testIsDefault() throws Exception {
    assertTrue(group.isDefault(Label.parseAbsolute("//pkg:foo")));
    assertFalse(group.isDefault(Label.parseAbsolute("//pkg:bar")));
    assertFalse(group.isDefault(Label.parseAbsolute("//pkg:baz")));
    assertFalse(group.isDefault(Label.parseAbsolute("//pkg:not_in_group")));
  }

  @Test
  public void testFulfillers() throws Exception {
    assertThat(group.getFulfillers(Label.parseAbsolute("//pkg:baz")))
        .containsExactly(Label.parseAbsolute("//pkg:foo"), Label.parseAbsolute("//pkg:bar"));
    assertThat(group.getFulfillers(Label.parseAbsolute("//pkg:bar")))
        .containsExactly(Label.parseAbsolute("//pkg:foo"));
    assertThat(group.getFulfillers(Label.parseAbsolute("//pkg:foo"))).isEmpty();
  }
}
