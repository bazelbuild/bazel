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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.packages.Type;
import java.util.List;

/**
 * Filters resources based on their qualifiers.
 *
 * <p>This includes filtering resources based on both the "resource_configuration_filters" and
 * "densities" attributes.
 *
 * <p>Whenever a new field is added to this class, be sure to add it to the {@link #equals(Object)}
 * and {@link #hashCode()} methods. Failure to do so isn't just bad practice; it could seriously
 * interfere with Bazel's caching performance.
 *
 * @deprecated Use "resource_configuration_filters" and "densities" directly.
 */
@Deprecated
public class ResourceFilterFactory {
  public static final String RESOURCE_CONFIGURATION_FILTERS_NAME = "resource_configuration_filters";
  public static final String DENSITIES_NAME = "densities";

  /**
   * The value of the {@link #RESOURCE_CONFIGURATION_FILTERS_NAME} attribute, as a list of qualifier
   * strings.
   */
  private final ImmutableList<String> configFilters;

  /** The value of the {@link #DENSITIES_NAME} attribute, as a list of qualifier strings. */
  private final ImmutableList<String> densities;

  /**
   * Constructor.
   *
   * @param configFilters the resource configuration filters, as a list of strings.
   * @param densities the density filters, as a list of strings.
   */
  ResourceFilterFactory(ImmutableList<String> configFilters, ImmutableList<String> densities) {
    this.configFilters = configFilters;
    this.densities = densities;
  }

  private static List<String> rawFiltersFromAttrs(AttributeMap attrs, String attrName) {
    if (attrs.isAttributeValueExplicitlySpecified(attrName)) {
      List<String> rawValue = attrs.get(attrName, Type.STRING_LIST);
      if (rawValue != null) {
        return rawValue;
      }
    }
    return ImmutableList.of();
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
  private static ImmutableList<String> extractFilters(List<String> rawValues) {
    if (rawValues.isEmpty()) {
      return ImmutableList.of();
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

  static ResourceFilterFactory fromRuleContextAndAttrs(RuleContext ruleContext)
      throws RuleErrorException {
    Preconditions.checkNotNull(ruleContext);

    if (!ruleContext.isLegalFragment(AndroidConfiguration.class)) {
      return empty();
    }

    return fromAttrs(ruleContext.attributes());
  }

  @VisibleForTesting
  static ResourceFilterFactory fromAttrs(AttributeMap attrs) {
    return from(
        rawFiltersFromAttrs(attrs, RESOURCE_CONFIGURATION_FILTERS_NAME),
        rawFiltersFromAttrs(attrs, DENSITIES_NAME));
  }

  static ResourceFilterFactory from(List<String> configFilters, List<String> densities) {
    if (configFilters.isEmpty() && densities.isEmpty()) {
      return empty();
    }

    return new ResourceFilterFactory(extractFilters(configFilters), extractFilters(densities));
  }

  @VisibleForTesting
  static ResourceFilterFactory empty() {
    return new ResourceFilterFactory(ImmutableList.of(), ImmutableList.of());
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
      AndroidResources localResources) {
    return ResourceFilter.empty();
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
    return configFilters.equals(other.configFilters) && densities.equals(other.densities);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(configFilters, densities);
  }
}
