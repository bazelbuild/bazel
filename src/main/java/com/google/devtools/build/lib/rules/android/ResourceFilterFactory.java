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
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/**
 * Filters resources based on their qualifiers.
 *
 * <p>This includes filtering resources based on both the "resource_configuration_filters" and
 * "densities" attributes.
 *
 * <p>Whenever a new field is added to this class, be sure to add it to the {@link #equals(Object)}
 * and {@link #hashCode()} methods. Failure to do so isn't just bad practice; it could seriously
 * interfere with Bazel's caching performance.
 */
public class ResourceFilterFactory {
  public static final String RESOURCE_CONFIGURATION_FILTERS_NAME = "resource_configuration_filters";
  public static final String DENSITIES_NAME = "densities";

  /**
   * Locales used for pseudolocation.
   *
   * <p>These are special resources that can be used to test how apps handles special cases (such as
   * particularly long text, accents, or left-to-right text). These resources are not provided like
   * other resources; instead, when the appropriate filters are passed in, aapt generates them based
   * on the default resources.
   *
   * <p>When these locales are specified in the configuration filters, even if we are filtering in
   * analysis, we need to pass *all* configuration filters to aapt - the pseudolocalization filters
   * themselves to trigger pseudolocalization, and the other filters to prevent aapt from filtering
   * matching resources out.
   */
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
  ResourceFilterFactory(
      ImmutableList<String> configFilters,
      ImmutableList<String> densities,
      FilterBehavior filterBehavior) {
    this.configFilters = configFilters;
    this.densities = densities;
    this.filterBehavior = filterBehavior;
  }

  private static boolean hasAttr(AttributeMap attrs, String attrName) {
    if (!attrs.isAttributeValueExplicitlySpecified(attrName)) {
      return false;
    }

    List<String> values = attrs.get(attrName, Type.STRING_LIST);
    return values != null && !values.isEmpty();
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

    // Use an ImmutableSet to remove duplicate values
    ImmutableSet.Builder<String> builder = ImmutableSet.builder();

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

    // Create a sorted copy so that ResourceFilterFactory objects with the same filters are treated
    // the same regardless of the ordering of those filters.
    return ImmutableList.sortedCopyOf(builder.build());
  }

  static ResourceFilterFactory fromRuleContext(RuleContext ruleContext) throws RuleErrorException {
    Preconditions.checkNotNull(ruleContext);

    if (!ruleContext.isLegalFragment(AndroidConfiguration.class)) {
      return empty();
    }

    ResourceFilterFactory base;
    if (AndroidAaptVersion.chooseTargetAaptVersion(ruleContext) == AndroidAaptVersion.AAPT2) {
      // aapt2 must have access to all of the resources in execution, so don't filter in analysis.
      base = empty(FilterBehavior.FILTER_IN_EXECUTION);
    } else {
      base = ruleContext.getFragment(AndroidConfiguration.class).getResourceFilterFactory();
    }

    return forBaseAndAttrs(base, ruleContext.attributes());
  }

  @VisibleForTesting
  static ResourceFilterFactory forBaseAndAttrs(ResourceFilterFactory base, AttributeMap attrs) {
    return base.withAttrsFrom(attrs);
  }

  /**
   * Creates a new {@link ResourceFilterFactory} based on this object's properties, overridden by
   * any filters specified in the passed {@link AttributeMap}.
   *
   * <p>A new object will always be returned, as returning the same object across multiple rules (as
   * would be done with {@link FilterBehavior#FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION}) causes
   * problems.
   */
  ResourceFilterFactory withAttrsFrom(AttributeMap attrs) {
    if (!hasFilters(attrs)) {
      return this;
    }

    return new ResourceFilterFactory(
        extractFilters(attrs, RESOURCE_CONFIGURATION_FILTERS_NAME),
        extractFilters(attrs, DENSITIES_NAME),
        filterBehavior);
  }

  ResourceFilterFactory withoutDynamicConfiguration() {
    if (!usesDynamicConfiguration()) {
      return this;
    }

    return empty(FilterBehavior.FILTER_IN_ANALYSIS);
  }

  private ImmutableList<FolderConfiguration> getConfigurationFilters(
      RuleErrorConsumer ruleErrorConsumer) {
    ImmutableList.Builder<FolderConfiguration> filterBuilder = ImmutableList.builder();
    for (String filter : configFilters) {
      addIfNotNull(
          getFolderConfiguration(ruleErrorConsumer, filter),
          filter,
          filterBuilder,
          ruleErrorConsumer,
          RESOURCE_CONFIGURATION_FILTERS_NAME);
    }

    return filterBuilder.build();
  }

  private FolderConfiguration getFolderConfiguration(
      RuleErrorConsumer ruleErrorConsumer, String filter) {

    // Clean up deprecated representations of resource qualifiers that FolderConfiguration can't
    // handle.
    for (DeprecatedQualifierHandler handler : deprecatedQualifierHandlers) {
      filter = handler.fixAttributeIfNeeded(ruleErrorConsumer, filter);
    }

    return FolderConfiguration.getConfigForQualifierString(filter);
  }

  private static final class DeprecatedQualifierHandler {
    private final Pattern pattern;
    private final String replacement;
    private final String description;

    private boolean warnedForAttribute = false;
    private boolean warnedForResources = false;

    private DeprecatedQualifierHandler(String pattern, String replacement, String description) {
      this.pattern = Pattern.compile(pattern);
      this.replacement = replacement;
      this.description = description;
    }

    private String fixAttributeIfNeeded(RuleErrorConsumer ruleErrorConsumer, String qualifier) {
      Matcher matcher = pattern.matcher(qualifier);

      if (!matcher.matches()) {
        return qualifier;
      }

      String fixed = matcher.replaceFirst(replacement);
      // We don't want to spam users. Only warn about this kind of issue once per target.
      // TODO(asteinb): Will this cause problems when settings are propagated via dynamic
      // configuration?
      if (!warnedForAttribute) {
        ruleErrorConsumer.attributeWarning(
            RESOURCE_CONFIGURATION_FILTERS_NAME,
            String.format(
                "When referring to %s, use of qualifier '%s' is deprecated. Use '%s' instead.",
                description, matcher.group(), fixed));
        warnedForAttribute = true;
      }
      return fixed;
    }

    private String fixResourceIfNeeded(
        RuleErrorConsumer ruleErrorConsumer, String qualifier, String resourceFolder) {
      Matcher matcher = pattern.matcher(qualifier);

      if (!matcher.matches()) {
        return qualifier;
      }

      String fixed = matcher.replaceFirst(replacement);

      // We don't want to spam users. Only warn about this kind of issue once per target.
      // TODO(asteinb): Will this cause problems when settings are propagated via dynamic
      // configuration?
      if (!warnedForResources) {
        warnedForResources = true;

        ruleErrorConsumer.ruleWarning(
            String.format(
                "For resource folder %s, when referring to %s, use of qualifier '%s' is deprecated."
                    + " Use '%s' instead.",
                resourceFolder, description, matcher.group(), fixed));
      }

      return fixed;
    }
  }

  /** List of deprecated qualifiers that should currently by handled with a warning */
  private final List<DeprecatedQualifierHandler> deprecatedQualifierHandlers =
      ImmutableList.of(
          /*
           * Aapt used to expect locale configurations of form 'en_US'. It now also supports the
           * correct 'en-rUS' format. For backwards comparability, use a regex to convert filters
           * with locales in the old format to filters with locales of the correct format.
           *
           * The correct format for locales is defined at
           * https://developer.android.com/guide/topics/resources/providing-resources.html#LocaleQualifier
           *
           * TODO(bazel-team): Migrate consumers away from the old Aapt locale format, then remove
           * this replacement.
           *
           * The regex is a bit complicated to avoid modifying potential new qualifiers that contain
           * underscores. Specifically, it searches for the entire beginning of the resource
           * qualifier, including (optionally) MCC and MNC, and then the locale itself.
           */
          new DeprecatedQualifierHandler(
              "^((mcc[0-9]{3}-(mnc[0-9]{3}-)?)?[a-z]{2})_([A-Z]{2}).*",
              "$1-r$4", "locale qualifiers with regions"),
          new DeprecatedQualifierHandler(
              "sr[_\\-]r?Latn.*", "b+sr+Latn", "Serbian in Latin characters"),
          new DeprecatedQualifierHandler(
              "es[_\\-]419.*", "b+es+419", "Spanish for Latin America and the Caribbean"));

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

  static ResourceFilterFactory empty() {
    return empty(DEFAULT_BEHAVIOR);
  }

  @VisibleForTesting
  static ResourceFilterFactory empty(FilterBehavior filterBehavior) {
    return new ResourceFilterFactory(
        ImmutableList.<String>of(), ImmutableList.<String>of(), filterBehavior);
  }

  /**
   * Gets an {@link ResourceFilter} that can be used to filter collections of artifacts.
   *
   * <p>In density-based filtering, the presence of one resource can affect whether another is
   * accepted or rejected. As such, both local and dependent resources must be passed.
   */
  ResourceFilter getResourceFilter(
      RuleErrorConsumer ruleErrorConsumer,
      ResourceDependencies resourceDeps,
      LocalResourceContainer localResources)
      throws RuleErrorException {
    if (!isPrefiltering()) {
      return ResourceFilter.empty();
    }

    ImmutableList<FolderConfiguration> folderConfigs = getConfigurationFilters(ruleErrorConsumer);

    ImmutableSet.Builder<Artifact> keptArtifacts = ImmutableSet.builder();
    List<BestArtifactsForDensity> bestArtifactsForAllDensities = new ArrayList<>();
    for (Density density : getDensities(ruleErrorConsumer)) {
      bestArtifactsForAllDensities.add(new BestArtifactsForDensity(ruleErrorConsumer, density));
    }

    // Look at the local and transitive resources.
    // TODO(b/68265485): In FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION, this will collapse the
    // NestedSet of resources at each node of the build graph. Instead, we should only filter local
    // resources at each non-binary target, and then filter both local and transitive resources in
    // the binary.
    for (Artifact artifact :
        Iterables.concat(localResources.getResources(), resourceDeps.getTransitiveResources())) {
      FolderConfiguration config = getConfigForArtifact(ruleErrorConsumer, artifact);

      // aapt explicitly ignores the version qualifier; duplicate this behavior here.
      config.setVersionQualifier(VersionQualifier.getQualifier(""));

      if (!matchesConfigurationFilters(folderConfigs, config)) {
        continue;
      }

      if (!shouldFilterByDensity(artifact)) {
        keptArtifacts.add(artifact);
        continue;
      }

      for (BestArtifactsForDensity bestArtifactsForDensity : bestArtifactsForAllDensities) {
        bestArtifactsForDensity.maybeAddArtifact(artifact);
      }
    }

    for (BestArtifactsForDensity bestArtifactsForDensity : bestArtifactsForAllDensities) {
      keptArtifacts.addAll(bestArtifactsForDensity.get());
    }

    return ResourceFilter.of(
        keptArtifacts.build(),
        (artifact) -> {
          // This class needs to record any dependent resources that were filtered out so that
          // resource processing ignores references to them in symbols files of  dependencies.
          String parentDir = artifact.getPath().getParentDirectory().getBaseName();
          filteredResources.add(parentDir + "/" + artifact.getFilename());
        });
  }

  /**
   * Tracks the best artifacts for a desired density for each combination of filename and
   * non-density qualifiers.
   *
   * <p>Filtering resources from multiple targets at once means we may get multiple resources with
   * the same filename and qualifiers (including density) - we accept all of them with the best
   * density, and the unwanted ones will be removed during resource merging.
   */
  private class BestArtifactsForDensity {
    private final RuleErrorConsumer ruleErrorConsumer;
    private final Density desiredDensity;
    private final Multimap<String, Artifact> nameAndConfigurationToBestArtifacts =
        HashMultimap.create();

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
      // We might encounter resource conflicts (multiple resources with the same name but different
      // locations) - in that case, we might have multiple best artifacts. In that case, keep them
      // all, and resource conflicts should will resolved during merging in execution.
      config.setDensityQualifier(null);
      String nameAndConfiguration =
          Joiner.on('/').join(config.getUniqueKey(), artifact.getFilename());

      Collection<Artifact> currentBest =
          nameAndConfigurationToBestArtifacts.get(nameAndConfiguration);

      if (currentBest.isEmpty()) {
        nameAndConfigurationToBestArtifacts.put(nameAndConfiguration, artifact);
        return;
      }

      double affinity = computeAffinity(artifact);
      // All of the current best artifacts should have the same density, so get the affinity of an
      // arbitrary one.
      double currentAffinity = computeAffinity(currentBest.iterator().next());

      if (affinity == currentAffinity) {
        nameAndConfigurationToBestArtifacts.put(nameAndConfiguration, artifact);
      }

      if (affinity < currentAffinity) {
        nameAndConfigurationToBestArtifacts.removeAll(nameAndConfiguration);
        nameAndConfigurationToBestArtifacts.put(nameAndConfiguration, artifact);
      }
    }

    public Collection<Artifact> get() {
      return nameAndConfigurationToBestArtifacts.values();
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

  private FolderConfiguration getConfigForArtifact(
      RuleErrorConsumer ruleErrorConsumer, Artifact artifact) {
    String containingFolder = getContainingFolder(artifact);

    if (containingFolder.contains("-")) {
      String[] parts = containingFolder.split("-", 2);
      String prefix = parts[0];
      String qualifiers = parts[1];

      for (DeprecatedQualifierHandler handler : deprecatedQualifierHandlers) {
        qualifiers = handler.fixResourceIfNeeded(ruleErrorConsumer, qualifiers, containingFolder);
      }

      containingFolder = String.format("%s-%s", prefix, qualifiers);
    }

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
    return hasFilters() && filterBehavior != FilterBehavior.FILTER_IN_EXECUTION;
  }

  boolean hasFilters() {
    return hasConfigurationFilters() || hasDensities();
  }

  public String getOutputDirectorySuffix() {
    if (!hasFilters()) {
      return null;
    }

    return getConfigurationFilterString() + "_" + getDensityString();
  }

  boolean usesDynamicConfiguration() {
    return filterBehavior == FilterBehavior.FILTER_IN_ANALYSIS_WITH_DYNAMIC_CONFIGURATION;
  }

  /**
   * Gets a list of resource names that should be ignored by resource processing if they don't
   * exist.
   *
   * <p>A target might filter out some of its dependency's targets. However, those filtered targets
   * have already been built into symbols files. The filtered resources must be passed to resource
   * processing at execution time so the code knows to ignore resources that were filtered out.
   * Without this, resource processing code would see references to those resources in
   * dependencies's symbol files, but then be unable to follow those references or know whether they
   * were missing due to resource filtering or a bug.
   */
  ImmutableList<String> getResourcesToIgnoreInExecution() {
    return filteredResources.build().asList();
  }

  /**
   * {@inheritDoc}
   *
   * <p>ResourceFilterFactory requires an accurately overridden equals() method to work correctly
   * with Bazel's caching and dynamic configuration.
   */
  @Override
  public boolean equals(Object object) {
    if (!(object instanceof ResourceFilterFactory)) {
      return false;
    }

    ResourceFilterFactory other = (ResourceFilterFactory) object;

    return filterBehavior == other.filterBehavior
        && configFilters.equals(other.configFilters)
        && densities.equals(other.densities)
        && filteredResources.build().equals(other.filteredResources.build());
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(filterBehavior, configFilters, densities, filteredResources.build());
  }

  /**
   * Converts command line settings for the filter behavior into an empty {@link
   * ResourceFilterFactory} object.
   */
  public static final class Converter
      implements com.google.devtools.common.options.Converter<ResourceFilterFactory> {
    private final FilterBehavior.Converter filterEnumConverter = new FilterBehavior.Converter();

    @Override
    public ResourceFilterFactory convert(String input) throws OptionsParsingException {

      return empty(filterEnumConverter.convert(input));
    }

    @Override
    public String getTypeDescription() {
      return filterEnumConverter.getTypeDescription();
    }
  }

  // Transitions for dealing with dynamically configured resource filtering:

  @Nullable
  PatchTransition getTopLevelPatchTransition(String ruleClass, AttributeMap attrs) {
    if (!usesDynamicConfiguration()) {
      // We're not using dynamic configuration, so we don't need to make a transition
      return null;
    }

    if (!ruleClass.equals("android_binary") || !hasFilters(attrs)) {
      // This target doesn't specify any filtering settings, so dynamically configured resource
      // filtering would be a waste of time.
      // If the target's dependencies include android_binary targets, those dependencies might
      // specify filtering settings, but we don't apply them dynamically since the chances of
      // encountering differing settings (leading to splitting the build graph and poor overall
      // performance) are high.
      return REMOVE_DYNAMICALLY_CONFIGURED_RESOURCE_FILTERING_TRANSITION;
    }

    // Continue using dynamically configured resource filtering, and propagate this target's
    // filtering settings.
    return new AddDynamicallyConfiguredResourceFilteringTransition(attrs);
  }

  public static final PatchTransition REMOVE_DYNAMICALLY_CONFIGURED_RESOURCE_FILTERING_TRANSITION =
      new RemoveDynamicallyConfiguredResourceFilteringTransition();

  private static final class RemoveDynamicallyConfiguredResourceFilteringTransition
      extends BaseDynamicallyConfiguredResourceFilteringTransition {
    @Override
    ResourceFilterFactory getNewResourceFilter(ResourceFilterFactory oldResourceFilterFactory) {
      return oldResourceFilterFactory.withoutDynamicConfiguration();
    }
  }

  @VisibleForTesting
  static final class AddDynamicallyConfiguredResourceFilteringTransition
      extends BaseDynamicallyConfiguredResourceFilteringTransition {
    private final AttributeMap attrs;

    AddDynamicallyConfiguredResourceFilteringTransition(AttributeMap attrs) {
      this.attrs = attrs;
    }

    @Override
    ResourceFilterFactory getNewResourceFilter(ResourceFilterFactory oldResourceFilterFactory) {
      return oldResourceFilterFactory.withAttrsFrom(attrs);
    }

    @VisibleForTesting
    AttributeMap getAttrs() {
      return attrs;
    }
  }

  private abstract static class BaseDynamicallyConfiguredResourceFilteringTransition
      implements PatchTransition {
    @Override
    public BuildOptions apply(BuildOptions options) {
      BuildOptions newOptions = options.clone();

      AndroidConfiguration.Options androidOptions =
          newOptions.get(AndroidConfiguration.Options.class);
      androidOptions.resourceFilterFactory =
          getNewResourceFilter(androidOptions.resourceFilterFactory);

      return newOptions;
    }

    abstract ResourceFilterFactory getNewResourceFilter(
        ResourceFilterFactory oldResourceFilterFactory);
  }
}
