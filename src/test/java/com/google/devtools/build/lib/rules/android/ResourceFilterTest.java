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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.rules.android.ResourceFilter.FilterBehavior;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link ResourceFilter}. */
// TODO(asteinb): Test behavior not already covered in this test, and, when practical, move unit
// tests currently located in {@link AndroidBinaryTest} to this class instead.
@RunWith(JUnit4.class)
public class ResourceFilterTest extends ResourceTestBase {
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
        ImmutableList.<String>of(),
        ImmutableList.<String>of(),
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.<String>of());
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
        ImmutableList.<String>of(),
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
        ImmutableList.of(
            "values-sr/foo.xml",
            "values-b+sr+Latn/foo.xml",
            "values-sr-Latn/foo.xml",
            "values-sr-rLatn/foo.xml"),
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
            "values-sr/foo.xml",
            "values-b+sr+Latn/foo.xml",
            "values-b+sr+Cyrl/foo.xml",
            "values-sr-Latn/foo.xml",
            "values-sr-rLatn/foo.xml"));

    errorConsumer.assertNoAttributeWarnings(ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME);
  }

  @Test
  public void testFilterSpanishLatinAmericaAndCaribbean() {
    testFilter(
        ImmutableList.of("es-419", "es_419", "b+es+419"),
        ImmutableList.of(),
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("values-es/foo.xml", "values-b+es+419/foo.xml", "values-es-419/foo.xml"),
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
        ImmutableList.of(
            "values-es/foo.xml",
            "values-b+es+419/foo.xml",
            "values-es-rUS/foo.xml",
            "values-es-419/foo.xml"));

    errorConsumer.assertNoAttributeWarnings(ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME);
  }

  /**
   * Tests that filtering with deprecated resource qualifiers only warns once for each type of
   * problem in each resource filter object.
   */
  @Test
  public void testFilterDeprecatedQualifierFormatsAttributeWarnings() {
    ImmutableList<String> badQualifiers =
        ImmutableList.of("en_US", "sr-Latn", "es-419", "mcc310-en_US", "sr_rLatn", "es_419");

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
    errorConsumer.assertNoRuleWarnings();

    // Filtering again with this filter should not produce additional warnings
    filter.filter(errorConsumer, ImmutableList.of());
    errorConsumer.assertNoAttributeWarnings(ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME);
    errorConsumer.assertNoRuleWarnings();

    // Filtering with a new filter should produce warnings again, since it is working on a different
    // target
    filter =
        new ResourceFilter(badQualifiers, ImmutableList.of(), FilterBehavior.FILTER_IN_ANALYSIS);
    filter.filter(errorConsumer, ImmutableList.of());
    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .containsExactlyElementsIn(expectedWarnings);
    errorConsumer.assertNoRuleWarnings();
  }

  @Test
  public void testFilterDeprecatedQualifierFormatsRuleWarnings() {
    ImmutableList<Artifact> badResources =
        getResources(
            "values-es_US/foo.xml",
            "drawables-sr-Latn/foo.xml",
            "stylables-es-419/foo.xml",
            "values-mcc310-es_US/foo.xml",
            "values-sr_rLatn/foo.xml",
            "drawables-es_419/foo.xml");

    ImmutableList<String> expectedWarnings =
        ImmutableList.of(
            "For resource folder drawables-sr-Latn, when referring to Serbian in Latin characters, "
                + "use of qualifier 'sr-Latn' is deprecated. Use 'b+sr+Latn' instead.",
            "For resource folder stylables-es-419, when referring to Spanish for Latin America and "
                + "the Caribbean, use of qualifier 'es-419' is deprecated. Use 'b+es+419' instead.",
            "For resource folder values-es_US, when referring to locale qualifiers with regions, "
                + "use of qualifier 'es_US' is deprecated. Use 'es-rUS' instead.");

    ResourceFilter filter =
        new ResourceFilter(
            ImmutableList.of("en"), ImmutableList.of(), FilterBehavior.FILTER_IN_ANALYSIS);

    filter.filter(errorConsumer, badResources);
    assertThat(errorConsumer.getAndClearRuleWarnings()).containsExactlyElementsIn(expectedWarnings);
    errorConsumer.assertNoAttributeWarnings(ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME);

    // Filtering again with this filter should not produce additional warnings
    filter.filter(errorConsumer, badResources);
    errorConsumer.assertNoRuleWarnings();
    errorConsumer.assertNoAttributeWarnings(ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME);

    // Filtering with a new filter should produce warnings again, since it is working on a different
    // target
    filter =
        new ResourceFilter(
            ImmutableList.of("en"), ImmutableList.of(), FilterBehavior.FILTER_IN_ANALYSIS);
    filter.filter(errorConsumer, badResources);
    assertThat(errorConsumer.getAndClearRuleWarnings()).containsExactlyElementsIn(expectedWarnings);
    errorConsumer.assertNoAttributeWarnings(ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME);
  }

  private void testNoopFilter(
      ImmutableList<String> resourceConfigurationFilters,
      ImmutableList<String> densities,
      FilterBehavior filterBehavior,
      List<String> resources) {
    testFilter(
        resourceConfigurationFilters,
        densities,
        filterBehavior,
        resources,
        ImmutableList.<String>of());
  }

  private void testFilter(
      ImmutableList<String> resourceConfigurationFilters,
      ImmutableList<String> densities,
      FilterBehavior filterBehavior,
      List<String> resourcesToKeep,
      List<String> resourcesToDiscard) {
    List<Artifact> unexpectedResources = new ArrayList<>();
    for (String resource : resourcesToDiscard) {
      unexpectedResources.add(getResource(resource));
    }

    List<Artifact> expectedResources = new ArrayList<>();
    for (String resource : resourcesToKeep) {
      expectedResources.add(getResource(resource));
    }

    ImmutableList<Artifact> allArtifacts =
        ImmutableList.copyOf(Iterables.concat(expectedResources, unexpectedResources));
    ImmutableList<Artifact> filtered =
        new ResourceFilter(resourceConfigurationFilters, densities, filterBehavior)
            .filter(errorConsumer, allArtifacts);

    assertThat(filtered).containsExactlyElementsIn(expectedResources).inOrder();
  }
}
