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
import com.google.common.truth.BooleanSubject;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.android.ResourceFilterFactory.AddDynamicallyConfiguredResourceFilteringTransition;
import com.google.devtools.build.lib.rules.android.ResourceFilterFactory.FilterBehavior;
import com.google.devtools.build.lib.testutil.FakeAttributeMapper;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link ResourceFilterFactory}. */
// TODO(asteinb): Test behavior not already covered in this test, and, when practical, move unit
// tests currently located in {@link AndroidBinaryTest} to this class instead.
@RunWith(JUnit4.class)
public class ResourceFilterFactoryTest extends ResourceTestBase {

  private NestedSet<ResourceContainer> getResourceContainers(ImmutableList<Artifact>... resources)
      throws Exception {
    NestedSetBuilder<ResourceContainer> builder = NestedSetBuilder.naiveLinkOrder();
    for (ImmutableList<Artifact> resourceList : resources) {
      builder.add(getResourceContainer(resourceList));
    }
    return builder.build();
  }

  private ResourceContainer getResourceContainer(ImmutableList<Artifact> resources)
      throws Exception {
    // Get dummy objects for irrelevant values required by the builder.
    Artifact manifest = getResource("manifest");

    // Include a hashCode of the resources in the label. A hack in ResourceContainer currently means
    // that class has a limited hashCode method that doesn't take resources into account.
    // TODO(bazel-team): Remove this hack once that one no longer exists.
    Label label =
        Label.create(
            manifest.getOwnerLabel().getPackageName(), "resourceContainer_" + resources.hashCode());

    return ResourceContainer.builder()
        .setResources(resources)
        .setResourcesRoots(
            LocalResourceContainer.getResourceRoots(errorConsumer, resources, "resource_files"))
        .setLabel(label)
        .setManifestExported(false)
        .setManifest(manifest)
        .build();
  }

  @Test
  public void testFilterInExecution() throws Exception {
    testNoopFilter(
        "en",
        "hdpi",
        FilterBehavior.FILTER_IN_EXECUTION,
        ImmutableList.of(
            "values-en/foo.xml", "values/foo.xml", "values-hdpi/foo.png", "values-ldpi/foo.png"));
  }

  @Test
  public void testFilterEmpty() throws Exception {
    testNoopFilter("", "", FilterBehavior.FILTER_IN_ANALYSIS, ImmutableList.<String>of());
  }

  @Test
  public void testFilterDefaultAndNonDefault() throws Exception {
    testNoopFilter(
        "en",
        "xhdpi,xxhdpi",
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("drawable/ic_clear.xml", "drawable-v21/ic_clear.xml"));
  }

  /**
   * Tests that version qualifiers are ignored for both resource qualifier and density filtering.
   */
  @Test
  public void testFilterVersionIgnored() throws Exception {
    testNoopFilter(
        "v4",
        "hdpi",
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("drawable-hdpi-v4/foo.png", "drawable-hdpi-v11/foo.png"));
  }

  @Test
  public void testFilterByDensityPersistsOrdering() throws Exception {
    testFilter(
        "",
        "hdpi,ldpi",
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
  public void testFilterOldLanguageFormat() throws Exception {
    testFilter(
        "en_US",
        "",
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("values-en/foo.xml", "values-en-rUS/foo.xml"),
        ImmutableList.of("values-fr/foo.xml", "values-en-rCA/foo.xml"));

    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilterFactory.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .hasSize(1);
  }

  @Test
  public void testFilterOldLanguageFormatWithAdditionalQualifiers() throws Exception {
    testFilter(
        "en_US-ldrtl",
        "",
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("values-en/foo.xml", "values-en-rUS/foo.xml"),
        ImmutableList.of("values-fr/foo.xml", "values-en-rCA/foo.xml"));

    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilterFactory.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .hasSize(1);
  }

  @Test
  public void testFilterOldLanguageFormatWithMcc() throws Exception {
    testFilter(
        "mcc111-en_US",
        "",
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("values-en/foo.xml", "values-en-rUS/foo.xml"),
        ImmutableList.of("values-fr/foo.xml", "values-en-rCA/foo.xml"));

    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilterFactory.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .hasSize(1);
  }

  @Test
  public void testFilterOldLanguageFormatWithMccAndMnc() throws Exception {
    testFilter(
        "mcc111-mnc111-en_US",
        "",
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("values-en/foo.xml", "values-en-rUS/foo.xml"),
        ImmutableList.of("values-fr/foo.xml", "values-en-rCA/foo.xml"));

    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilterFactory.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .hasSize(1);
  }

  @Test
  public void testFilterSerbianLatinCharacters() throws Exception {
    testFilter(
        "sr-Latn,sr-rLatn,sr_Latn,b+sr+Latn",
        "",
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
                ResourceFilterFactory.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .hasSize(1);
  }

  @Test
  public void testFilterSerbianCharactersNotSpecified() throws Exception {
    testNoopFilter(
        "sr",
        "",
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of(
            "values-sr/foo.xml",
            "values-b+sr+Latn/foo.xml",
            "values-b+sr+Cyrl/foo.xml",
            "values-sr-Latn/foo.xml",
            "values-sr-rLatn/foo.xml"));

    errorConsumer.assertNoAttributeWarnings(
        ResourceFilterFactory.RESOURCE_CONFIGURATION_FILTERS_NAME);
  }

  @Test
  public void testFilterSpanishLatinAmericaAndCaribbean() throws Exception {
    testFilter(
        "es-419,es_419,b+es+419",
        "",
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of("values-es/foo.xml", "values-b+es+419/foo.xml", "values-es-419/foo.xml"),
        // Spanish with another region specified should be filtered out.
        ImmutableList.of("values-es-rUS/foo.xml"));

    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilterFactory.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .hasSize(1);
  }

  @Test
  public void testFilterSpanishRegionNotSpecified() throws Exception {
    testNoopFilter(
        "es",
        "",
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of(
            "values-es/foo.xml",
            "values-b+es+419/foo.xml",
            "values-es-rUS/foo.xml",
            "values-es-419/foo.xml"));

    errorConsumer.assertNoAttributeWarnings(
        ResourceFilterFactory.RESOURCE_CONFIGURATION_FILTERS_NAME);
  }

  /**
   * Tests that filtering with deprecated resource qualifiers only warns once for each type of
   * problem in each resource filter object.
   */
  @Test
  public void testFilterDeprecatedQualifierFormatsAttributeWarnings() throws RuleErrorException {
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

    ResourceFilterFactory filter =
        new ResourceFilterFactory(
            badQualifiers, ImmutableList.of(), FilterBehavior.FILTER_IN_ANALYSIS);

    doFilter(filter, ImmutableList.of());
    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilterFactory.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .containsExactlyElementsIn(expectedWarnings);
    errorConsumer.assertNoRuleWarnings();

    // Filtering again with this filter should not produce additional warnings
    doFilter(filter, ImmutableList.of());
    errorConsumer.assertNoAttributeWarnings(
        ResourceFilterFactory.RESOURCE_CONFIGURATION_FILTERS_NAME);
    errorConsumer.assertNoRuleWarnings();

    // Filtering with a new filter should produce warnings again, since it is working on a different
    // target
    filter =
        new ResourceFilterFactory(
            badQualifiers, ImmutableList.of(), FilterBehavior.FILTER_IN_ANALYSIS);
    doFilter(filter, ImmutableList.of());
    assertThat(
            errorConsumer.getAndClearAttributeWarnings(
                ResourceFilterFactory.RESOURCE_CONFIGURATION_FILTERS_NAME))
        .containsExactlyElementsIn(expectedWarnings);
    errorConsumer.assertNoRuleWarnings();
  }

  @Test
  public void testFilterDeprecatedQualifierFormatsRuleWarnings() throws RuleErrorException {
    ImmutableList<Artifact> badResources =
        getResources(
            "values-es_US/foo.xml",
            "drawable-sr-Latn/foo.xml",
            "layout-es-419/foo.xml",
            "values-mcc310-es_US/foo.xml",
            "values-sr_rLatn/foo.xml",
            "drawable-es_419/foo.xml");

    ImmutableList<String> expectedWarnings =
        ImmutableList.of(
            "For resource folder drawable-sr-Latn, when referring to Serbian in Latin characters, "
                + "use of qualifier 'sr-Latn' is deprecated. Use 'b+sr+Latn' instead.",
            "For resource folder layout-es-419, when referring to Spanish for Latin America and "
                + "the Caribbean, use of qualifier 'es-419' is deprecated. Use 'b+es+419' instead.",
            "For resource folder values-es_US, when referring to locale qualifiers with regions, "
                + "use of qualifier 'es_US' is deprecated. Use 'es-rUS' instead.");

    ResourceFilterFactory filter =
        new ResourceFilterFactory(
            ImmutableList.of("en"), ImmutableList.of(), FilterBehavior.FILTER_IN_ANALYSIS);

    doFilter(filter, badResources);
    assertThat(errorConsumer.getAndClearRuleWarnings()).containsExactlyElementsIn(expectedWarnings);
    errorConsumer.assertNoAttributeWarnings(
        ResourceFilterFactory.RESOURCE_CONFIGURATION_FILTERS_NAME);

    // Filtering again with this filter should not produce additional warnings
    doFilter(filter, badResources);
    errorConsumer.assertNoRuleWarnings();
    errorConsumer.assertNoAttributeWarnings(
        ResourceFilterFactory.RESOURCE_CONFIGURATION_FILTERS_NAME);

    // Filtering with a new filter should produce warnings again, since it is working on a different
    // target
    filter =
        new ResourceFilterFactory(
            ImmutableList.of("en"), ImmutableList.of(), FilterBehavior.FILTER_IN_ANALYSIS);
    doFilter(filter, badResources);
    assertThat(errorConsumer.getAndClearRuleWarnings()).containsExactlyElementsIn(expectedWarnings);
    errorConsumer.assertNoAttributeWarnings(
        ResourceFilterFactory.RESOURCE_CONFIGURATION_FILTERS_NAME);
  }

  @Test
  public void testFilterResourceConflict() throws Exception {
    testNoopFilter(
        "en",
        "hdpi",
        FilterBehavior.FILTER_IN_ANALYSIS,
        ImmutableList.of(
            "first-subdir/res/drawable-en-hdpi/foo.png",
            "second-subdir/res/drawable-en-hdpi/foo.png"));
  }

  @Test
  public void testFilterWithDynamicConfiguration() throws Exception {
    testFilter(
        "en",
        "hdpi",
        FilterBehavior.FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION,
        ImmutableList.of("drawable-en-hdpi/foo.png"),
        ImmutableList.of("drawable-en-ldpi/foo.png", "drawable-fr-hdpi/foo.png"));
  }

  private void testNoopFilter(
      String resourceConfigurationFilters,
      String densities,
      FilterBehavior filterBehavior,
      List<String> resources)
      throws Exception {
    testFilter(
        resourceConfigurationFilters,
        densities,
        filterBehavior,
        resources,
        ImmutableList.<String>of());
  }

  private void testFilter(
      String resourceConfigurationFilters,
      String densities,
      FilterBehavior filterBehavior,
      List<String> resourcesToKeep,
      List<String> resourcesToDiscard)
      throws Exception {
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
    ResourceFilterFactory resourceFilterFactory =
        makeResourceFilter(resourceConfigurationFilters, densities, filterBehavior);
    ImmutableList<Artifact> filtered = doFilter(resourceFilterFactory, allArtifacts);

    assertThat(filtered).containsExactlyElementsIn(expectedResources).inOrder();

    // Only dependencies need to be tracked for ignoring in execution
    assertThat(resourceFilterFactory.getResourcesToIgnoreInExecution()).isEmpty();
  }

  @Test
  public void testFilterLocalAndTransitive() throws Exception {
    Artifact localResourceToKeep = getResource("drawable-en-hdpi/local.png");
    Artifact localResourceToDiscard = getResource("drawable-en-ldpi/local.png");

    // These resources go in different ResourceContainers to ensure we are filter across all
    // resources.
    Artifact directResourceToKeep = getResource("direct/drawable-en-hdpi/direct.png");
    Artifact directResourceToDiscard = getResource("direct/drawable-en-ldpi/direct.png");
    Artifact transitiveResourceToKeep = getResource("transitive/drawable-en-hdpi/transitive.png");
    Artifact transitiveResourceToDiscard =
        getResource("transitive/drawable-en-ldpi/transitive.png");

    LocalResourceContainer localResources =
        LocalResourceContainer.forResources(
            errorConsumer, ImmutableList.of(localResourceToKeep, localResourceToDiscard));

    ResourceDependencies resourceDependencies =
        ResourceDependencies.empty()
            .withResources(
                getResourceContainers(
                    ImmutableList.of(transitiveResourceToDiscard),
                    ImmutableList.of(transitiveResourceToKeep)),
                getResourceContainers(
                    ImmutableList.of(directResourceToDiscard),
                    ImmutableList.of(directResourceToKeep)),
                new NestedSetBuilder<Artifact>(Order.NAIVE_LINK_ORDER)
                    .add(directResourceToDiscard)
                    .add(directResourceToKeep)
                    .addTransitive(
                        NestedSetBuilder.create(
                            Order.NAIVE_LINK_ORDER,
                            transitiveResourceToDiscard,
                            transitiveResourceToKeep))
                    .build());

    ResourceFilterFactory resourceFilterFactory =
        makeResourceFilter("en", "hdpi", FilterBehavior.FILTER_IN_ANALYSIS);
    ResourceFilter filter =
        resourceFilterFactory.getResourceFilter(
            errorConsumer, resourceDependencies, localResources);

    assertThat(localResources.filter(errorConsumer, filter).getResources())
        .containsExactly(localResourceToKeep);

    ResourceDependencies filteredResourceDeps = resourceDependencies.filter(filter);

    // TODO: Remove - assert was same order before
    assertThat(resourceDependencies.getTransitiveResources())
        .containsAllOf(directResourceToKeep, transitiveResourceToKeep)
        .inOrder();

    assertThat(filteredResourceDeps.getTransitiveResources())
        .containsExactly(directResourceToKeep, transitiveResourceToKeep)
        .inOrder();

    List<ResourceContainer> directContainers =
        filteredResourceDeps.getDirectResourceContainers().toList();
    assertThat(directContainers).hasSize(2);

    ResourceContainer directToDiscard = directContainers.get(0);
    assertThat(directToDiscard.getResources()).isEmpty();
    assertThat(directToDiscard.getResourcesRoots()).isEmpty();

    ResourceContainer directToKeep = directContainers.get(1);
    assertThat(directToKeep.getResources()).containsExactly(directResourceToKeep);
    assertThat(directToKeep.getResourcesRoots())
        .containsExactly(
            directResourceToKeep.getExecPath().getParentDirectory().getParentDirectory());

    List<ResourceContainer> transitiveContainers =
        filteredResourceDeps.getTransitiveResourceContainers().toList();
    assertThat(transitiveContainers).hasSize(2);

    ResourceContainer transitiveToDiscard = transitiveContainers.get(0);
    assertThat(transitiveToDiscard.getResources()).isEmpty();
    assertThat(transitiveToDiscard.getResourcesRoots()).isEmpty();

    ResourceContainer transitiveToKeep = transitiveContainers.get(1);
    assertThat(transitiveToKeep.getResources())
        .containsExactly(transitiveResourceToKeep);
    assertThat(transitiveToKeep.getResourcesRoots())
        .containsExactly(
            transitiveResourceToKeep
                .getExecPath()
                .getParentDirectory()
                .getParentDirectory());

    // We tell the resource processing actions to ignore references to filtered resources from
    // dependencies.
    assertThat(resourceFilterFactory.getResourcesToIgnoreInExecution())
        .containsExactly("drawable-en-ldpi/direct.png", "drawable-en-ldpi/transitive.png");
  }

  @Test
  public void testIsPrefilteringFilterInExecution() throws Exception {
    assertIsPrefiltering(FilterBehavior.FILTER_IN_EXECUTION, false);
  }

  @Test
  public void testIsPrefilteringFilterInAnalysis() throws Exception {
    assertIsPrefiltering(FilterBehavior.FILTER_IN_ANALYSIS, true);
  }

  @Test
  public void testIsPrefilteringFilterInAnalysisWithDynamicConfiguration() throws Exception {
    assertIsPrefiltering(FilterBehavior.FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION, true);
  }

  private void assertIsPrefiltering(FilterBehavior behavior, boolean expectWhenNonEmpty)
      throws Exception {
    // Empty filters should never prefilter
    assertIsPrefiltering(false, false, behavior).isFalse();

    // Prefiltering behavior should be the same regardless of which setting is set
    assertIsPrefiltering(true, false, behavior).isEqualTo(expectWhenNonEmpty);
    assertIsPrefiltering(false, true, behavior).isEqualTo(expectWhenNonEmpty);
    assertIsPrefiltering(true, true, behavior).isEqualTo(expectWhenNonEmpty);
  }

  private BooleanSubject assertIsPrefiltering(
      boolean hasConfigurationFilters, boolean hasDensities, FilterBehavior behavior)
      throws Exception {
    return assertThat(
        makeResourceFilter(
                hasConfigurationFilters ? "en" : "", hasDensities ? "hdpi" : "", behavior)
            .isPrefiltering());
  }

  @Test
  public void testGetOutputDirectorySuffixEmpty() throws Exception {
    assertThat(
            makeResourceFilter("", "", FilterBehavior.FILTER_IN_ANALYSIS)
                .getOutputDirectorySuffix())
        .isNull();
  }

  @Test
  public void testGetOutputDirectoryOnlyFilterConfigurations() throws Exception {
    String configurationFilters = "en,es-rUS,fr";
    assertThat(
            makeResourceFilter(configurationFilters, "", FilterBehavior.FILTER_IN_ANALYSIS)
                .getOutputDirectorySuffix())
        .isEqualTo(configurationFilters + "_");
  }

  @Test
  public void testGetOutputDirectoryOnlyDensities() throws Exception {
    String densities = "hdpi,ldpi,xhdpi";
    assertThat(
            makeResourceFilter("", densities, FilterBehavior.FILTER_IN_ANALYSIS)
                .getOutputDirectorySuffix())
        .isEqualTo("_" + densities);
  }

  /**
   * Only densities is a legal (if unhelpful) resource_configuration_filters settings. Ensure it
   * produces a different output directory than a similar densities value.
   */
  @Test
  public void testGetOutputDirectoryDensitiesAreDifferentFromDensityConfigurationFilters()
      throws Exception {
    ResourceFilterFactory configurationFilter =
        makeResourceFilter("hdpi", "", FilterBehavior.FILTER_IN_ANALYSIS);
    ResourceFilterFactory densityFilter =
        makeResourceFilter("", "hdpi", FilterBehavior.FILTER_IN_ANALYSIS);

    assertThat(configurationFilter.getOutputDirectorySuffix())
        .isNotEqualTo(densityFilter.getOutputDirectorySuffix());
  }

  @Test
  public void testGetOutputDirectory() throws Exception {
    assertThat(
            makeResourceFilter("en,fr-rCA", "hdpi,ldpi", FilterBehavior.FILTER_IN_ANALYSIS)
                .getOutputDirectorySuffix())
        .isEqualTo("en,fr-rCA_hdpi,ldpi");
  }

  /**
   * Asserts that identical but differently ordered arguments still produce the same output
   * directory. If filters are identical, there's no reason to rebuild.
   */
  @Test
  public void testGetOutputDirectoryDifferentlyOrdered() throws Exception {
    ResourceFilterFactory first =
        makeResourceFilter("en,fr", "hdpi,ldpi", FilterBehavior.FILTER_IN_ANALYSIS);
    ResourceFilterFactory second =
        makeResourceFilter("fr,en", "ldpi,hdpi", FilterBehavior.FILTER_IN_ANALYSIS);
    assertThat(first.getOutputDirectorySuffix()).isEqualTo(second.getOutputDirectorySuffix());
  }

  @Test
  public void testGetOutputDirectoryDuplicated() throws Exception {
    ResourceFilterFactory duplicated =
        makeResourceFilter("en,en", "hdpi,hdpi", FilterBehavior.FILTER_IN_ANALYSIS);
    ResourceFilterFactory normal =
        makeResourceFilter("en", "hdpi", FilterBehavior.FILTER_IN_ANALYSIS);

    assertThat(duplicated.getOutputDirectorySuffix()).isEqualTo(normal.getOutputDirectorySuffix());
  }

  private ResourceFilterFactory makeResourceFilter(
      String resourceConfigurationFilters, String densities, FilterBehavior behavior)
      throws Exception {
    return makeResourceFilter(
        ImmutableList.of(resourceConfigurationFilters), ImmutableList.of(densities), behavior);
  }

  private ResourceFilterFactory makeResourceFilter(
      ImmutableList<String> resourceConfigurationFilters,
      ImmutableList<String> densities,
      FilterBehavior behavior)
      throws Exception {

    return ResourceFilterFactory.forBaseAndAttrs(
        ResourceFilterFactory.empty(behavior),
        getAttributeMap(resourceConfigurationFilters, densities));
  }

  private AttributeMap getAttributeMap(
      ImmutableList<String> resourceConfigurationFilters, ImmutableList<String> densities) {
    return FakeAttributeMapper.builder()
        .withStringList(
            ResourceFilterFactory.RESOURCE_CONFIGURATION_FILTERS_NAME, resourceConfigurationFilters)
        .withStringList(ResourceFilterFactory.DENSITIES_NAME, densities)
        .build();
  }

  private ImmutableList<Artifact> doFilter(
      ResourceFilterFactory resourceFilterFactory, ImmutableList<Artifact> artifacts)
      throws RuleErrorException {
    LocalResourceContainer localResourceContainer =
        LocalResourceContainer.forResources(errorConsumer, artifacts);

    ResourceDependencies resourceDeps = ResourceDependencies.empty();

    ResourceFilter filter =
        resourceFilterFactory.getResourceFilter(
            errorConsumer, resourceDeps, localResourceContainer);

    assertThat(resourceDeps.filter(filter)).isSameAs(resourceDeps);

    return localResourceContainer.filter(errorConsumer, filter).getResources();
  }

  @Test
  public void testWithAttrsFromAttrsNotSpecified() throws Exception {
    assertThat(
            ResourceFilterFactory.forBaseAndAttrs(
                    ResourceFilterFactory.empty(FilterBehavior.FILTER_IN_ANALYSIS),
                    FakeAttributeMapper.empty())
                .hasFilters())
        .isFalse();
  }

  @Test
  public void testGetTopLevelTransitionFilterInExecution() throws Exception {
    assertThat(
            getTopLevelTransition(
                ImmutableList.of("en"),
                ImmutableList.of("hdpi"),
                FilterBehavior.FILTER_IN_EXECUTION,
                true))
        .isNull();
  }

  @Test
  public void testGetTopLevelTransitionFilterInAnalysis() throws Exception {
    assertThat(
            getTopLevelTransition(
                ImmutableList.of("en"),
                ImmutableList.of("hdpi"),
                FilterBehavior.FILTER_IN_ANALYSIS,
                true))
        .isNull();
  }

  @Test
  public void testGetTopLevelTransitionNotBinary() throws Exception {
    assertThat(
            getTopLevelTransition(
                ImmutableList.of("en"),
                ImmutableList.of("hdpi"),
                FilterBehavior.FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION,
                false))
        .isSameAs(
            ResourceFilterFactory.REMOVE_DYNAMICALLY_CONFIGURED_RESOURCE_FILTERING_TRANSITION);
  }

  @Test
  public void testGetTopLevelTransitionNoFilters() throws Exception {
    assertThat(
            getTopLevelTransition(
                ImmutableList.<String>of(),
                ImmutableList.<String>of(),
                FilterBehavior.FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION,
                true))
        .isSameAs(
            ResourceFilterFactory.REMOVE_DYNAMICALLY_CONFIGURED_RESOURCE_FILTERING_TRANSITION);
  }

  @Test
  public void testGetTopLevelTransition() throws Exception {
    ImmutableList<String> resourceConfigurationFilters = ImmutableList.of("en");
    ImmutableList<String> densities = ImmutableList.of("hdpi");
    PatchTransition transition =
        getTopLevelTransition(
            resourceConfigurationFilters,
            densities,
            FilterBehavior.FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION,
            true);

    assertThat(transition).isInstanceOf(AddDynamicallyConfiguredResourceFilteringTransition.class);

    AddDynamicallyConfiguredResourceFilteringTransition addTransition =
        (AddDynamicallyConfiguredResourceFilteringTransition) transition;
    ResourceFilterFactory foundFilter =
        ResourceFilterFactory.forBaseAndAttrs(
            ResourceFilterFactory.empty(
                FilterBehavior.FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION),
            addTransition.getAttrs());

    ResourceFilterFactory expectedFilter =
        makeResourceFilter(
            resourceConfigurationFilters,
            densities,
            FilterBehavior.FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION);
    assertThat(foundFilter).isEqualTo(expectedFilter);
  }

  private PatchTransition getTopLevelTransition(
      ImmutableList<String> resourceConfigurationFilters,
      ImmutableList<String> densities,
      FilterBehavior behavior,
      boolean isBinary)
      throws Exception {
    AttributeMap attrs = getAttributeMap(resourceConfigurationFilters, densities);
    return makeResourceFilter("", "", behavior)
        .getTopLevelPatchTransition(isBinary ? "android_binary" : "android_library", attrs);
  }

  @Test
  public void testRemoveDynamicConfigurationTransition() throws Exception {
    assertPatchTransition(
        makeResourceFilter(
            "en", "ldpi", FilterBehavior.FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION),
        ResourceFilterFactory.REMOVE_DYNAMICALLY_CONFIGURED_RESOURCE_FILTERING_TRANSITION,
        ResourceFilterFactory.empty(FilterBehavior.FILTER_IN_ANALYSIS));
  }

  @Test
  public void testAddDynamicConfigurationTransitionDynamicConfiguration() throws Exception {
    ImmutableList<String> resourceConfigurationFilters = ImmutableList.of("en", "es-rUS", "fr");
    ImmutableList<String> densities = ImmutableList.of("ldpi", "hdpi");

    AttributeMap attrs = getAttributeMap(resourceConfigurationFilters, densities);

    assertPatchTransition(
        ResourceFilterFactory.empty(FilterBehavior.FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION),
        new ResourceFilterFactory.AddDynamicallyConfiguredResourceFilteringTransition(attrs),
        makeResourceFilter(
            resourceConfigurationFilters,
            densities,
            FilterBehavior.FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION));
  }

  private void assertPatchTransition(
      ResourceFilterFactory oldResourceFilterFactory,
      PatchTransition transition,
      ResourceFilterFactory expectedNewResourceFilterFactory) {
    AndroidConfiguration.Options oldAndroidOptions = getAndroidOptions(oldResourceFilterFactory);

    BuildOptions oldOptions = BuildOptions.builder().add(oldAndroidOptions).build();
    BuildOptions newOptions = transition.apply(oldOptions);

    // The old options should not have been changed
    assertThat(oldAndroidOptions.resourceFilterFactory).isSameAs(oldResourceFilterFactory);
    assertThat(oldAndroidOptions).isEqualTo(getAndroidOptions(oldResourceFilterFactory));

    // Besides the ResourceFilterFactory, the new options should be the same as the old ones
    assertThat(newOptions.getOptions()).hasSize(1);
    AndroidConfiguration.Options newAndroidOptions =
        newOptions.get(AndroidConfiguration.Options.class);
    assertThat(newAndroidOptions).isEqualTo(getAndroidOptions(expectedNewResourceFilterFactory));
  }

  private AndroidConfiguration.Options getAndroidOptions(
      ResourceFilterFactory resourceFilterFactory) {
    AndroidConfiguration.Options androidOptions =
        (AndroidConfiguration.Options) new AndroidConfiguration.Options().getDefault();
    androidOptions.resourceFilterFactory = resourceFilterFactory;

    return androidOptions;
  }
}
