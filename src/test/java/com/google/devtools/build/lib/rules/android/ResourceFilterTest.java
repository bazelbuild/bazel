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

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.android.ResourceFilter.FilterBehavior;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link ResourceFilter}. */
// TODO(asteinb): Test behavior not already covered in this test, and, when practical, move unit
// tests currently located in {@link AndroidBinaryTest} to this class instead.
@RunWith(JUnit4.class)
public class ResourceFilterTest {
  private FakeRuleErrorConsumer errorConsumer;

  @Before
  public void setup() {
    errorConsumer = new FakeRuleErrorConsumer();
  }

  private static final class FakeRuleErrorConsumer implements RuleErrorConsumer {
    // Use an ArrayListMultimap since it allows duplicates - we'll want to know if a warning is
    // reported twice.
    private final Multimap<String, String> attributeWarnings = ArrayListMultimap.create();

    @Override
    public void ruleWarning(String message) {}

    @Override
    public void ruleError(String message) {
      assertWithMessage(message).fail();
    }

    @Override
    public void attributeWarning(String attrName, String message) {
      attributeWarnings.put(attrName, message);
    }

    @Override
    public void attributeError(String attrName, String message) {
      assertWithMessage(message + " (attribute: " + attrName + ")").fail();
    }

    public Collection<String> getAndClearAttributeWarnings(String attrName) {
      if (!attributeWarnings.containsKey(attrName)) {
        return ImmutableList.of();
      }

      return attributeWarnings.removeAll(attrName);
    }

    public void assertNoAttributeWarnings(String attrName) {
      assertThat(attributeWarnings).doesNotContainKey(attrName);
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

  @Test
  public void testFilterByDensityPersistsOrdering() {
    testFilter(
        ImmutableList.of(),
        ImmutableList.of("hdpi", "ldpi"),
        FilterBehavior.FILTER_IN_ANALYSIS,
        // If we add resources to the output list in density order, these resources will be
        // rearranged.
        ImmutableList.of(
            "drawable-hdpi/foo.png",
            "drawable-ldpi/foo.png",
            "drawable-ldpi/bar.png",
            "drawable-hdpi/bar.png"),
        // Filter out some resources to make sure the original list isn't just returned because the
        // filtering was a no-op.
        ImmutableList.of("drawable-mdpi/foo.png", "drawable-mdpi/bar.png"));
  }
  /** Tests handling of Aapt's old region format */
  @Test
  public void testFilterOldLanguageFormat() {
    testFilter(
        ImmutableList.of("en_US"),
        ImmutableList.of(),
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("values-en/foo.xml", "values-en-rUS/foo.xml"),
        ImmutableList.of("values-fr/foo.xml", "values-en-rCA/foo.xml"));

    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .hasSize(1);
  }

  @Test
  public void testFilterOldLanguageFormatWithAdditionalQualifiers() {
    testFilter(
        ImmutableList.of("en_US-ldrtl"),
        ImmutableList.of(),
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("values-en/foo.xml", "values-en-rUS/foo.xml"),
        ImmutableList.of("values-fr/foo.xml", "values-en-rCA/foo.xml"));

    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .hasSize(1);
  }

  @Test
  public void testFilterOldLanguageFormatWithMcc() {
    testFilter(
        ImmutableList.of("mcc111-en_US"),
        ImmutableList.of(),
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("values-en/foo.xml", "values-en-rUS/foo.xml"),
        ImmutableList.of("values-fr/foo.xml", "values-en-rCA/foo.xml"));

    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .hasSize(1);
  }

  @Test
  public void testFilterOldLanguageFormatWithMccAndMnc() {
    testFilter(
        ImmutableList.of("mcc111-mnc111-en_US"),
        ImmutableList.of(),
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("values-en/foo.xml", "values-en-rUS/foo.xml"),
        ImmutableList.of("values-fr/foo.xml", "values-en-rCA/foo.xml"));

    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .hasSize(1);
  }

  @Test
  public void testFilterSerbianLatinCharacters() {
    testFilter(
        ImmutableList.of("sr-Latn", "sr-rLatn", "sr_Latn", "b+sr+Latn"),
        ImmutableList.of(),
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("values-sr/foo.xml", "values-b+sr+Latn/foo.xml"),
        // Serbian in explicitly Cyrillic characters should be filtered out.
        ImmutableList.of("values-b+sr+Cyrl/foo.xml"));

    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .hasSize(1);
  }

  @Test
  public void testFilterSerbianCharactersNotSpecified() {
    testNoopFilter(
        ImmutableList.of("sr"),
        ImmutableList.of(),
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of(
            "values-sr/foo.xml", "values-b+sr+Latn/foo.xml", "values-b+sr+Cyrl/foo.xml"));

    errorConsumer.assertNoAttributeWarnings(ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME);
  }

  @Test
  public void testFilterSpanishLatinAmericaAndCaribbean() {
    testFilter(
        ImmutableList.of("es-419", "es_419", "b+es+419"),
        ImmutableList.of(),
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("values-es/foo.xml", "values-b+es+419/foo.xml"),
        // Spanish with another region specified should be filtered out.
        ImmutableList.of("values-es-rUS/foo.xml"));

    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .hasSize(1);
  }

  @Test
  public void testFilterSpanishRegionNotSpecified() {
    testNoopFilter(
        ImmutableList.of("es"),
        ImmutableList.of(),
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("values-es/foo.xml", "values-b+es+419/foo.xml", "values-es-rUS/foo.xml"));

    errorConsumer.assertNoAttributeWarnings(ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME);
  }

  /**
   * Tests that filtering with deprecated resource qualifiers only warns once for each type of
   * problem in each resource filter object.
   */
  @Test
  public void testFilterDeprecatedQualifierFormatsWarnings() {
    ImmutableList<String> badQualifiers =
        ImmutableList.of("en_US", "sr-Latn", "es-419", "mcc310-en_US", "sr_Latn", "es_419");

    ImmutableList<String> expectedWarnings =
        ImmutableList.of(
            "When referring to Serbian in Latin characters, use of qualifier 'sr-Latn' is"
                + " deprecated. Use 'b+sr+Latn' instead.",
            "When referring to Spanish for Latin America and the Caribbean, use of qualifier"
                + " 'es-419' is deprecated. Use 'b+es+419' instead.",
            "When referring to locale qualifiers with regions, use of qualifier 'en_US' is"
                + " deprecated. Use 'en-rUS' instead.");

    ResourceFilter filter =
        new ResourceFilter(badQualifiers, ImmutableList.of(), FilterBehavior.FILTER_IN_ANALYSIS);

    filter.filter(errorConsumer, ImmutableList.of());
    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .containsExactlyElementsIn(expectedWarnings);

    // Filtering again with this filter should not produce additional warnings
    filter.filter(errorConsumer, ImmutableList.of());
    errorConsumer.assertNoAttributeWarnings(ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME);

    // Filtering with a new filter should produce warnings again, since it is working on a different
    // target
    filter =
        new ResourceFilter(badQualifiers, ImmutableList.of(), FilterBehavior.FILTER_IN_ANALYSIS);
    filter.filter(errorConsumer, ImmutableList.of());
    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .containsExactlyElementsIn(expectedWarnings);
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
            .filter(errorConsumer, allArtifacts);

    assertThat(filtered).containsExactlyElementsIn(expectedResources).inOrder();
  }

  private static Artifact getArtifact(String pathString) {
    return new Artifact(FILE_SYSTEM.getPath("/java/android/res/" + pathString), ROOT);
  }
}
