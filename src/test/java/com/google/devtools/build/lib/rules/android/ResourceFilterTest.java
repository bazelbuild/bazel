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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.android.ResourceFilter.FilterBehavior;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link ResourceFilter}. */
// TODO(asteinb): Test behavior not already covered in this test, and, when practical, move unit
// tests currently located in {@link AndroidBinaryTest} to this class instead.
@RunWith(JUnit4.class)
public class ResourceFilterTest {
  public static final RuleErrorConsumer FAKE_ERROR_CONSUMER = new FakeRuleErrorConsumer();

  private static final class FakeRuleErrorConsumer implements RuleErrorConsumer {
    @Override
    public void ruleWarning(String message) {}

    @Override
    public void ruleError(String message) {
      assertWithMessage(message).fail();
    }

    @Override
    public void attributeWarning(String attrName, String message) {}

    @Override
    public void attributeError(String attrName, String message) {
      assertWithMessage(message + " (attribute: " + attrName + ")").fail();
    }
  };

  private static final FileSystem FILE_SYSTEM = new InMemoryFileSystem();
  private static final Root ROOT = Root.asSourceRoot(FILE_SYSTEM.getRootDirectory());

  @Test
  public void testFilterInExecution() {
    testNoopFilter(
        ImmutableList.of("en"),
        ImmutableList.of("hdpi"),
        FilterBehavior.FILTER_IN_EXECUTION,
        ImmutableList.of(
            "values-en/foo.xml", "values/foo.xml", "values-hdpi/foo.png", "values-ldpi/foo.png"));
  }

  @Test
  public void testFilterEmpty() {
    testNoopFilter(
        ImmutableList.of(),
        ImmutableList.of(),
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of());
  }

  @Test
  public void testFilterDefaultAndNonDefault() {
    testNoopFilter(
        ImmutableList.of("en"),
        ImmutableList.of("xhdpi", "xxhdpi"),
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("drawable/ic_clear.xml", "drawable-v21/ic_clear.xml"));
  }

  /**
   * Tests that version qualifiers are ignored for both resource qualifier and density filtering.
   */
  @Test
  public void testFilterVersionIgnored() {
    testNoopFilter(
        ImmutableList.of("v4"),
        ImmutableList.of("hdpi"),
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("drawable-hdpi-v4/foo.png", "drawable-hdpi-v11/foo.png"));
  }

  private void testNoopFilter(
      ImmutableList<String> resourceConfigurationFilters,
      ImmutableList<String> densities,
      FilterBehavior filterBehavior,
      List<String> resources) {
    testFilter(
        resourceConfigurationFilters, densities, filterBehavior, resources, ImmutableList.of());
  }

  private void testFilter(
      ImmutableList<String> resourceConfigurationFilters,
      ImmutableList<String> densities,
      FilterBehavior filterBehavior,
      List<String> resourcesToKeep,
      List<String> resourcesToDiscard) {
    List<Artifact> unexpectedResources = new ArrayList<>();
    for (String resource : resourcesToDiscard) {
      unexpectedResources.add(getArtifact(resource));
    }

    List<Artifact> expectedResources = new ArrayList<>();
    for (String resource : resourcesToKeep) {
      expectedResources.add(getArtifact(resource));
    }

    ImmutableList<Artifact> allArtifacts =
        ImmutableList.copyOf(Iterables.concat(expectedResources, unexpectedResources));
    ImmutableList<Artifact> filtered =
        new ResourceFilter(resourceConfigurationFilters, densities, filterBehavior)
            .filter(FAKE_ERROR_CONSUMER, allArtifacts);

    assertThat(filtered).containsExactlyElementsIn(expectedResources);
  }

  private static Artifact getArtifact(String pathString) {
    return new Artifact(FILE_SYSTEM.getPath("/java/android/res/" + pathString), ROOT);
  }
}
