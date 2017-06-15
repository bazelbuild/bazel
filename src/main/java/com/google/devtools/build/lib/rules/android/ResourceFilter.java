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
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.OptionsParsingException;
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

  @VisibleForTesting
  static enum FilterBehavior {
    /**
     * Resources will be filtered in execution. This class will just pass the filtering parameters
     * to the appropriate resource processing actions.
     */
    FILTER_IN_EXECUTION,

    /**
     * Resources will be filtered in analysis. In android_binary targets, all resources will be
     * filtered by this class, and only resources that are accepted will be passed to resource
     * processing actions.
     */
    FILTER_IN_ANALYSIS,

    /**
     * Resources will be filtered in each android target in analysis. Filter settings will be
     * extracted from android_binary targets and passed to all their dependencies using dynamic
     * configuration. Only resources that are accepted by filtering will be passed to resource
     * processing actions or to reverse dependencies.
     */
    FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION;

    private static final class Converter extends EnumConverter<FilterBehavior> {
      Converter() {
        super(FilterBehavior.class, "resource filter behavior");
      }
    }
  }

  static final FilterBehavior DEFAULT_BEHAVIOR = FilterBehavior.FILTER_IN_EXECUTION;

  /**
   * The value of the {@link #RESOURCE_CONFIGURATION_FILTERS_NAME} attribute, as a list of qualifier
   * strings.
   */
  private final ImmutableList<String> configFilters;

  /** The value of the {@link #DENSITIES_NAME} attribute, as a list of qualifier strings. */
  private final ImmutableList<String> densities;

  /** A builder for a set of strings representing resources that were filtered using this class. */
  private final ImmutableSet.Builder<String> filteredResources = ImmutableSet.builder();

  private final FilterBehavior filterBehavior;

  /**
   * Constructor.
   *
   * @param configFilters the resource configuration filters, as a list of strings.
   * @param densities the density filters, as a list of strings.
   * @param filterBehavior the behavior of this filter.
   */
  @VisibleForTesting
  ResourceFilter(
      ImmutableList<String> configFilters,
      ImmutableList<String> densities,
      FilterBehavior filterBehavior) {
    this.configFilters = configFilters;
    this.densities = densities;
    this.filterBehavior = filterBehavior;
  }

  private static boolean hasAttr(AttributeMap attrs, String attrName) {
    return attrs.isAttributeValueExplicitlySpecified(attrName);
  }

  static boolean hasFilters(RuleContext ruleContext) {
    return hasFilters(ruleContext.attributes());
  }

  static boolean hasFilters(AttributeMap attrs) {
    return hasAttr(attrs, RESOURCE_CONFIGURATION_FILTERS_NAME) || hasAttr(attrs, DENSITIES_NAME);
  }

  /**
   * Extracts filters from an AttributeMap, as a list of strings.
   *
   * <p>In BUILD files, string lists can be represented as a list of strings, a single
   * comma-separated string, or a combination of both. This method outputs a single list of
   * individual string values, which can then be passed directly to resource processing actions.
   *
   * @return the values of this attribute contained in the {@link AttributeMap}, as a list.
   */
  private static ImmutableList<String> extractFilters(AttributeMap attrs, String attrName) {
    if (!hasAttr(attrs, attrName)) {
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
    List<String> rawValues = attrs.get(attrName, Type.STRING_LIST);
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

    if (!ruleContext.isLegalFragment(AndroidConfiguration.class)) {
      return empty(DEFAULT_BEHAVIOR);
    }

    return ruleContext
        .getFragment(AndroidConfiguration.class)
        .getResourceFilter()
        .withAttrsFrom(ruleContext.attributes());
  }

  /**
   * Creates a new {@link ResourceFilter} based on this object's properties, overridden by any
   * filters specified in the passed {@link AttributeMap}.
   *
   * <p>A new object will always be returned, as returning the same object across multiple rules (as
   * would be done with {@link FilterBehavior#FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION}) causes
   * problems.
   */
  ResourceFilter withAttrsFrom(AttributeMap attrs) {
    if (!hasFilters(attrs)) {
      return new ResourceFilter(configFilters, densities, filterBehavior);
    }

    return new ResourceFilter(
        extractFilters(attrs, RESOURCE_CONFIGURATION_FILTERS_NAME),
        extractFilters(attrs, DENSITIES_NAME),
        filterBehavior);
  }

  private ImmutableList<FolderConfiguration> getConfigurationFilters(
      RuleErrorConsumer ruleErrorConsumer) {
    ImmutableList.Builder<FolderConfiguration> filterBuilder = ImmutableList.builder();
    for (String filter : configFilters) {
      addIfNotNull(
          getFolderConfiguration(filter),
          filter,
          filterBuilder,
          ruleErrorConsumer,
          RESOURCE_CONFIGURATION_FILTERS_NAME);
    }

    return filterBuilder.build();
  }

  private static FolderConfiguration getFolderConfiguration(String filter) {
    /*
     * Aapt used to expect locale configurations of form 'en_US'. It now also supports the correct
     * 'en-rUS' format. For backwards comparability, use a regex to convert filters with locales in
     * the old format to filters with locales of the correct format.
     *
     * The correct format for locales is defined at
     * https://developer.android.com/guide/topics/resources/providing-resources.html#LocaleQualifier
     *
     * TODO(bazel-team): Migrate consumers away from the old Aapt locale format, then remove this
     * replacement.
     *
     * The regex is a bit complicated to avoid modifying potential new qualifiers that contain
     * underscores. Specifically, it searches for the entire beginning of the resource qualifier,
     * including (optionally) MCC and MNC, and then the locale itself.
     */
    String fixedFilter =
        filter.replaceFirst("^((mcc[0-9]{3}-(mnc[0-9]{3}-)?)?[a-z]{2})_([A-Z]{2})", "$1-r$4");
    return FolderConfiguration.getConfigForQualifierString(fixedFilter);
  }

  private ImmutableList<Density> getDensities(RuleErrorConsumer ruleErrorConsumer) {
    ImmutableList.Builder<Density> densityBuilder = ImmutableList.builder();
    for (String density : densities) {
      addIfNotNull(
          Density.getEnum(density), density, densityBuilder, ruleErrorConsumer, DENSITIES_NAME);
    }

    return densityBuilder.build();
  }

  /** Reports an attribute error if the given item is null, and otherwise adds it to the builder. */
  private static <T> void addIfNotNull(
      T item,
      String itemString,
      ImmutableList.Builder<T> builder,
      RuleErrorConsumer ruleErrorConsumer,
      String attrName) {
    if (item == null) {
      ruleErrorConsumer.attributeError(
          attrName, "String '" + itemString + "' is not a valid value for " + attrName);
    } else {
      builder.add(item);
    }
  }

  static ResourceFilter empty(RuleContext ruleContext) {
    return empty(fromRuleContext(ruleContext).filterBehavior);
  }

  private static ResourceFilter empty(FilterBehavior filterBehavior) {
    return new ResourceFilter(
        ImmutableList.<String>of(), ImmutableList.<String>of(), filterBehavior);
  }

  /**
   * Filters a NestedSet of resource containers that contain dependencies of the current rule. This
   * may be a no-op if this filter is empty or if resource prefiltering is disabled.
   */
  NestedSet<ResourceContainer> filterDependencies(
      RuleErrorConsumer ruleErrorConsumer, NestedSet<ResourceContainer> resources) {
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
      builder.add(resource.filter(ruleErrorConsumer, this));
    }

    return builder.build();
  }

  ImmutableList<Artifact> filter(
      RuleErrorConsumer ruleErrorConsumer, ImmutableList<Artifact> artifacts) {
    if (!isPrefiltering()) {
      return artifacts;
    }

    /*
     * Build an ImmutableSet rather than an ImmutableList to remove duplicate Artifacts in the case
     * where one Artifact is the best option for multiple densities.
     */
    ImmutableSet.Builder<Artifact> builder = ImmutableSet.builder();

    List<BestArtifactsForDensity> bestArtifactsForAllDensities = new ArrayList<>();
    for (Density density : getDensities(ruleErrorConsumer)) {
      bestArtifactsForAllDensities.add(new BestArtifactsForDensity(ruleErrorConsumer, density));
    }

    ImmutableList<FolderConfiguration> folderConfigs = getConfigurationFilters(ruleErrorConsumer);

    for (Artifact artifact : artifacts) {
      FolderConfiguration config = getConfigForArtifact(ruleErrorConsumer, artifact);

      // aapt explicitly ignores the version qualifier; duplicate this behavior here.
      config.setVersionQualifier(VersionQualifier.getQualifier(""));

      if (!matchesConfigurationFilters(folderConfigs, config)) {
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

    // TODO(asteinb): We should only build a new list if some artifacts were filtered out. If
    // nothing was filtered, we can be more efficient by returning the original list instead.
    return keptArtifacts.asList();
  }

  /**
   * Tracks the best artifact for a desired density for each combination of filename and non-density
   * qualifiers.
   */
  private static class BestArtifactsForDensity {
    private final RuleErrorConsumer ruleErrorConsumer;
    private final Density desiredDensity;

    // Use a LinkedHashMap to preserve determinism.
    private final Map<String, Artifact> nameAndConfigurationToBestArtifact = new LinkedHashMap<>();

    public BestArtifactsForDensity(RuleErrorConsumer ruleErrorConsumer, Density density) {
      this.ruleErrorConsumer = ruleErrorConsumer;
      desiredDensity = density;
    }

    /**
     * @param artifact if this artifact is a better match for this object's desired density than any
     *     other artifacts with the same name and non-density configuration, adds it to this object.
     */
    public void maybeAddArtifact(Artifact artifact) {
      FolderConfiguration config = getConfigForArtifact(ruleErrorConsumer, artifact);

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
      DensityQualifier resourceQualifier =
          getConfigForArtifact(ruleErrorConsumer, artifact).getDensityQualifier();
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

  private static FolderConfiguration getConfigForArtifact(
      RuleErrorConsumer ruleErrorConsumer, Artifact artifact) {
    String containingFolder = getContainingFolder(artifact);
    FolderConfiguration config = FolderConfiguration.getConfigForFolder(containingFolder);

    if (config == null) {
      ruleErrorConsumer.ruleError(
          "Resource folder '" + containingFolder + "' has invalid resource qualifiers");

      return FolderConfiguration.getConfigForQualifierString("");
    }

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

  private static boolean matchesConfigurationFilters(
      ImmutableList<FolderConfiguration> folderConfigs, FolderConfiguration config) {
    for (FolderConfiguration filter : folderConfigs) {
      if (config.isMatchFor(filter)) {
        return true;
      }
    }

    return folderConfigs.isEmpty();
  }

  /**
   * Returns if this object contains a non-empty resource configuration filter.
   *
   * <p>Note that non-empty filters are not guaranteed to filter resources during the analysis
   * phase.
   */
  boolean hasConfigurationFilters() {
    return !configFilters.isEmpty();
  }

  String getConfigurationFilterString() {
    return Joiner.on(',').join(configFilters);
  }

  /**
   * Returns if this object contains a non-empty density filter.
   *
   * <p>Note that non-empty filters are not guaranteed to filter resources during the analysis
   * phase.
   */
  boolean hasDensities() {
    return !densities.isEmpty();
  }

  String getDensityString() {
    return Joiner.on(',').join(densities);
  }

  List<String> getDensities() {
    return densities;
  }

  boolean isPrefiltering() {
    return filterBehavior == FilterBehavior.FILTER_IN_ANALYSIS
        || filterBehavior == FilterBehavior.FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION;
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

  /**
   * Converts command line settings for the filter behavior into an empty {@link ResourceFilter}
   * object.
   */
  public static final class Converter
      implements com.google.devtools.common.options.Converter<ResourceFilter> {
    private final FilterBehavior.Converter filterEnumConverter = new FilterBehavior.Converter();

    @Override
    public ResourceFilter convert(String input) throws OptionsParsingException {

      return ResourceFilter.empty(filterEnumConverter.convert(input));
    }

    @Override
    public String getTypeDescription() {
      return filterEnumConverter.getTypeDescription();
    }
  }
}
