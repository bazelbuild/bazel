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

import com.android.ide.common.resources.configuration.DensityQualifier;
import com.android.ide.common.resources.configuration.FolderConfiguration;
import com.android.ide.common.resources.configuration.VersionQualifier;
import com.android.resources.Density;
import com.android.resources.ResourceFolderType;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.syntax.Type;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Filters resources based on their qualifiers.
 *
 * <p>This includes filtering resources based on both the "resource_configuration_filters" and
 * "densities" attributes.
 */
public class ResourceFilter {
  public static final String RESOURCE_CONFIGURATION_FILTERS_NAME = "resource_configuration_filters";
  public static final String DENSITIES_NAME = "densities";

  /** The current {@link RuleContext}, used for reporting errors. */
  private final RuleContext ruleContext;

  /**
   * A list of configuration filters that should be applied during the analysis phase. If resources
   * should not be filtered in analysis (e.g., if android_binary.prefilter_resources is set to 0),
   * this list should be empty.
   */
  private final ImmutableList<FolderConfiguration> configurationFilters;

  /**
   * The value of the {@link #RESOURCE_CONFIGURATION_FILTERS_NAME} attribute, as a list of qualifier
   * strings. This value is passed directly as input to resource processing utilities that run in
   * the execution phase, so it may represent different qualifiers than those in {@link
   * #configurationFilters} if resources should not be pre-filtered during analysis.
   */
  private final ImmutableList<String> configurationFilterStrings;

  /**
   * A list of density filters that should be applied during the analysis phase. If resources should
   * not be filtered in analysis (e.g., if prefilter_resources = 0), this list should be empty.
   */
  private final ImmutableList<Density> densities;

  /**
   * The value of the {@link #DENSITIES_NAME} attribute, as a list of qualifier strings. This value
   * is passed directly as input to resource processing utilities that run in the execution phase,
   * so it may represent different qualifiers than those in {@link #densities} if resources should
   * not be pre-filtered during analysis.
   */
  private final ImmutableList<String> densityStrings;

  private final ImmutableSet.Builder<String> filteredResources = ImmutableSet.builder();

  /**
   * Constructor.
   *
   * @param ruleContext the current {@link RuleContext}, used for reporting errors
   * @param configurationFilters a list of configuration filters that should be applied during
   *     analysis time. If resources should not be filtered in analysis (e.g., if
   *     prefilter_resources = 0), this list should be empty.
   * @param configurationFilterStrings the resource configuration filters, as a list of strings.
   * @param densities a list of densities that should be applied to filter resources during analysis
   *     time. If resources should not be filtered in analysis (e.g., if prefilter_resources = 0),
   *     this list should be empty.
   * @param densityStrings the densities, as a list of strings.
   */
  private ResourceFilter(
      RuleContext ruleContext,
      ImmutableList<FolderConfiguration> configurationFilters,
      ImmutableList<String> configurationFilterStrings,
      ImmutableList<Density> densities,
      ImmutableList<String> densityStrings) {
    this.ruleContext = ruleContext;
    this.configurationFilters = configurationFilters;
    this.configurationFilterStrings = configurationFilterStrings;
    this.densities = densities;
    this.densityStrings = densityStrings;
  }

  private static boolean hasAttr(RuleContext ruleContext, String attrName) {
    return ruleContext.attributes().isAttributeValueExplicitlySpecified(attrName);
  }

  static boolean hasFilters(RuleContext ruleContext) {
    return hasAttr(ruleContext, RESOURCE_CONFIGURATION_FILTERS_NAME)
        || hasAttr(ruleContext, DENSITIES_NAME);
  }

  /**
   * Extracts filters from the current RuleContext, as a list of strings.
   *
   * <p>In BUILD files, string lists can be represented as a list of strings, a single
   * comma-separated string, or a combination of both. This method outputs a single list of
   * individual string values, which can then be passed directly to resource processing actions.
   *
   * @return the values of this attribute contained in the {@link RuleContext}, as a list.
   */
  private static ImmutableList<String> extractFilters(RuleContext ruleContext, String attrName) {
    if (!hasAttr(ruleContext, attrName)) {
      return ImmutableList.<String>of();
    }

    /*
     * To deal with edge cases involving placement of whitespace and multiple strings inside a
     * single item of the given list, manually build the list here rather than call something like
     * {@link RuleContext#getTokenizedStringListAttr}.
     *
     * Filter out all empty values, even those that were explicitly provided. Paying attention to
     * empty values is never helpful: even if code handled them correctly (and not all of it does)
     * empty filter values result in all resources matching the empty filter, meaning that filtering
     * does nothing (even if non-empty filters were also provided).
     */
    List<String> rawValues = ruleContext.attributes().get(attrName, Type.STRING_LIST);
    ImmutableList.Builder<String> builder = ImmutableList.builder();

    for (String rawValue : rawValues) {
      if (rawValue.contains(",")) {
        for (String token : rawValue.split(",")) {
          if (!token.trim().isEmpty()) {
            builder.add(token.trim());
          }
        }
      } else if (!rawValue.isEmpty()) {
        builder.add(rawValue);
      }
    }

    return builder.build();
  }

  static ResourceFilter fromRuleContext(RuleContext ruleContext) {
    Preconditions.checkNotNull(ruleContext);

    if (!hasFilters(ruleContext)) {
      return empty(ruleContext);
    }

    ImmutableList<String> resourceConfigurationFilters =
        extractFilters(ruleContext, RESOURCE_CONFIGURATION_FILTERS_NAME);
    ImmutableList<String> densities = extractFilters(ruleContext, DENSITIES_NAME);

    boolean usePrefiltering =
        ruleContext.getFragment(AndroidConfiguration.class).useResourcePrefiltering();

    ImmutableList.Builder<FolderConfiguration> filterBuilder = ImmutableList.builder();
    if (usePrefiltering) {
      for (String filter : resourceConfigurationFilters) {
        addIfNotNull(
            FolderConfiguration.getConfigForQualifierString(filter),
            filter,
            filterBuilder,
            ruleContext,
            RESOURCE_CONFIGURATION_FILTERS_NAME);
      }
    }

    ImmutableList.Builder<Density> densityBuilder = ImmutableList.builder();
    if (usePrefiltering) {
      for (String density : densities) {
        addIfNotNull(
            Density.getEnum(density), density, densityBuilder, ruleContext, DENSITIES_NAME);
      }
    }

    return new ResourceFilter(
        ruleContext,
        filterBuilder.build(),
        resourceConfigurationFilters,
        densityBuilder.build(),
        densities);
  }

  /** Reports an attribute error if the given item is null, and otherwise adds it to the builder. */
  private static <T> void addIfNotNull(
      T item,
      String itemString,
      ImmutableList.Builder<T> builder,
      RuleContext ruleContext,
      String attrName) {
    if (item == null) {
      ruleContext.attributeError(
          attrName, "String '" + itemString + "' is not a valid value for " + attrName);
    } else {
      builder.add(item);
    }
  }

  static ResourceFilter empty(RuleContext ruleContext) {
    return new ResourceFilter(
        ruleContext,
        ImmutableList.<FolderConfiguration>of(),
        ImmutableList.<String>of(),
        ImmutableList.<Density>of(),
        ImmutableList.<String>of());
  }

  /**
   * Filters a NestedSet of resource containers. This may be a no-op if this filter is empty or if
   * resource prefiltering is disabled.
   */
  NestedSet<ResourceContainer> filter(NestedSet<ResourceContainer> resources) {
    if (!isPrefiltering()) {
      /*
       * If the filter is empty or resource prefiltering is disabled, just return the original,
       * rather than make a copy.
       *
       * Resources should only be prefiltered in top-level android targets (such as android_binary).
       * The output of resource processing, which includes the input NestedSet<ResourceContainer>
       * returned by this method, is exposed to other actions via the AndroidResourcesProvider. If
       * this method did a no-op copy and collapse in those cases, rather than just return the
       * original NestedSet, we would lose all of the advantages around memory and time that
       * NestedSets provide: each android_library target would have to copy the resources provided
       * by its dependencies into a new NestedSet rather than just create a NestedSet pointing at
       * its dependencies's NestedSets.
       */
      return resources;
    }

    NestedSetBuilder<ResourceContainer> builder = new NestedSetBuilder<>(resources.getOrder());

    for (ResourceContainer resource : resources) {
      builder.add(resource.filter(this));
    }

    return builder.build();
  }

  ImmutableList<Artifact> filter(ImmutableList<Artifact> artifacts) {
    if (!isPrefiltering()) {
      return artifacts;
    }

    /*
     * Build an ImmutableSet rather than an ImmutableList to remove duplicate Artifacts in the case
     * where one Artifact is the best option for multiple densities.
     */
    ImmutableSet.Builder<Artifact> builder = ImmutableSet.builder();

    List<BestArtifactsForDensity> bestArtifactsForAllDensities = new ArrayList<>();
    for (Density density : densities) {
      bestArtifactsForAllDensities.add(new BestArtifactsForDensity(density));
    }

    for (Artifact artifact : artifacts) {
      FolderConfiguration configuration = getConfigForArtifact(artifact);
      if (!matchesConfigurationFilters(configuration)) {
        continue;
      }

      if (!shouldFilterByDensity(artifact)) {
        builder.add(artifact);
        continue;
      }

      for (BestArtifactsForDensity bestArtifactsForDensity : bestArtifactsForAllDensities) {
        bestArtifactsForDensity.maybeAddArtifact(artifact);
      }
    }

    for (BestArtifactsForDensity bestArtifactsForDensity : bestArtifactsForAllDensities) {
      builder.addAll(bestArtifactsForDensity.get());
    }

    ImmutableSet<Artifact> keptArtifacts = builder.build();
    for (Artifact artifact : artifacts) {
      if (keptArtifacts.contains(artifact)) {
        continue;
      }

      String parentDir = artifact.getPath().getParentDirectory().getBaseName();
      filteredResources.add(parentDir + "/" + artifact.getFilename());
    }

    return keptArtifacts.asList();
  }

  /**
   * Tracks the best artifact for a desired density for each combination of filename and non-density
   * qualifiers.
   */
  private class BestArtifactsForDensity {
    private final Density desiredDensity;

    // Use a LinkedHashMap to preserve determinism.
    private final Map<String, Artifact> nameAndConfigurationToBestArtifact = new LinkedHashMap<>();

    public BestArtifactsForDensity(Density density) {
      desiredDensity = density;
    }

    /**
     * @param artifact if this artifact is a better match for this object's desired density than any
     *     other artifacts with the same name and non-density configuration, adds it to this object.
     */
    public void maybeAddArtifact(Artifact artifact) {
      FolderConfiguration config = getConfigForArtifact(artifact);

      // We want to find a single best artifact for each combination of non-density qualifiers and
      // filename. Combine those two values to create a single unique key.
      config.setDensityQualifier(null);
      String nameAndConfiguration = config.getUniqueKey() + "/" + artifact.getFilename();

      Artifact currentBest = nameAndConfigurationToBestArtifact.get(nameAndConfiguration);

      if (currentBest == null || computeAffinity(artifact) < computeAffinity(currentBest)) {
        nameAndConfigurationToBestArtifact.put(nameAndConfiguration, artifact);
      }
    }

    /** @return the collection of best Artifacts for this density. */
    public Collection<Artifact> get() {
      return nameAndConfigurationToBestArtifact.values();
    }

    /**
     * Compute how well this artifact matches the {@link #desiredDensity}.
     *
     * <p>Various different codebases have different and sometimes contradictory methods for which
     * resources are better in different situations. All of them agree that an exact match is best,
     * but:
     *
     * <p>The android common code (see {@link FolderConfiguration#getDensityQualifier()} treats
     * larger densities as better than non-matching smaller densities.
     *
     * <p>aapt code to filter assets by density prefers the smallest density that is larger than or
     * the same as the desired density, or, lacking that, the largest available density.
     *
     * <p>Other implementations of density filtering include Gradle (to filter which resources
     * actually get built into apps) and Android code itself (for the device to decide which
     * resource to use).
     *
     * <p>This particular implementation is based on {@link
     * com.google.devtools.build.android.DensitySpecificResourceFilter}, which filters resources by
     * density during execution. It prefers to use exact matches when possible, then tries to find
     * resources with exactly double the desired density for particularly easy downsizing, and
     * otherwise prefers resources that are closest to the desired density, relative to the smaller
     * of the available and desired densities.
     *
     * <p>Once we always filter resources during analysis, we should be able to completely remove
     * that code.
     *
     * @return a score for how well the artifact matches. Lower scores indicate better matches.
     */
    private double computeAffinity(Artifact artifact) {
      DensityQualifier resourceQualifier = getConfigForArtifact(artifact).getDensityQualifier();
      if (resourceQualifier == null) {
        return Double.MAX_VALUE;
      }

      int resourceDensity = resourceQualifier.getValue().getDpiValue();
      int density = desiredDensity.getDpiValue();

      if (resourceDensity == density) {
        // Exact match is the best.
        return -2;
      }

      if (resourceDensity == 2 * density) {
        // It's very efficient to downsample an image that's exactly twice the screen
        // density, so we prefer that over other non-perfect matches.
        return -1;
      }

      // Find the ratio between the larger and smaller of the available and desired densities.
      double densityRatio =
          Math.max(density, resourceDensity) / (double) Math.min(density, resourceDensity);

      if (density < resourceDensity) {
        return densityRatio;
      }

      // Apply a slight bias against resources that are smaller than those of the desired density.
      // This becomes relevant only when we are considering multiple resources with the same ratio.
      return densityRatio + 0.01;
    }
  }

  private FolderConfiguration getConfigForArtifact(Artifact artifact) {
    String containingFolder = getContainingFolder(artifact);
    FolderConfiguration config = FolderConfiguration.getConfigForFolder(containingFolder);

    if (config == null) {
      ruleContext.ruleError(
          "Resource folder '" + containingFolder + "' has invalid resource qualifiers");

      return FolderConfiguration.getConfigForQualifierString("");
    }

    // aapt explicitly ignores the version qualifier; duplicate this behavior here.
    config.setVersionQualifier(VersionQualifier.getQualifier(""));

    return config;
  }

  /**
   * Checks if we should filter this artifact by its density.
   *
   * <p>We filter by density if there are densities to filter by, the artifact is in a Drawable
   * directory, and the artifact is not an XML file.
   *
   * <p>Similarly-named XML files may contain different resource definitions, so it's impossible to
   * ensure that all required resources will be provided without that XML file unless we parse it.
   */
  private boolean shouldFilterByDensity(Artifact artifact) {
    return !densities.isEmpty()
        && !artifact.getExtension().equals("xml")
        && ResourceFolderType.getFolderType(getContainingFolder(artifact))
            == ResourceFolderType.DRAWABLE;
  }

  private static String getContainingFolder(Artifact artifact) {
    return artifact.getPath().getParentDirectory().getBaseName();
  }

  private boolean matchesConfigurationFilters(FolderConfiguration config) {
    for (FolderConfiguration filter : configurationFilters) {
      if (config.isMatchFor(filter)) {
        return true;
      }
    }

    return configurationFilters.isEmpty();
  }

  /**
   * Returns if this object contains a non-empty resource configuration filter.
   *
   * <p>Note that non-empty filters are not guaranteed to filter resources during the analysis
   * phase.
   */
  boolean hasConfigurationFilters() {
    return !configurationFilterStrings.isEmpty();
  }

  String getConfigurationFilterString() {
    return Joiner.on(',').join(configurationFilterStrings);
  }

  /**
   * Returns if this object contains a non-empty density filter.
   *
   * <p>Note that non-empty filters are not guaranteed to filter resources during the analysis
   * phase.
   */
  boolean hasDensities() {
    return !densityStrings.isEmpty();
  }

  String getDensityString() {
    return Joiner.on(',').join(densityStrings);
  }

  List<String> getDensities() {
    return densityStrings;
  }

  boolean isPrefiltering() {
    return !densities.isEmpty() || !configurationFilters.isEmpty();
  }

  /**
   * TODO: Stop tracking these once android_library targets also filter resources correctly.
   *
   * <p>Currently, android_library targets pass do no filtering, and all resources are built into
   * their symbol files. The android_binary target filters out these resources in analysis. However,
   * the filtered resources must be passed to resource processing at execution time so the code
   * knows to ignore resources that were filtered out. Without this, resource processing code would
   * see references to those resources in dependencies's symbol files, but then be unable to follow
   * those references or know whether they were missing due to resource filtering or a bug.
   *
   * @return a list of resources that were filtered out by this filter
   */
  ImmutableList<String> getFilteredResources() {
    return filteredResources.build().asList();
  }
}
