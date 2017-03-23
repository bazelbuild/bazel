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

import com.android.ide.common.resources.configuration.FolderConfiguration;
import com.android.ide.common.resources.configuration.VersionQualifier;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;

/** Filters resources based on their qualifiers. */
public class ResourceConfigurationFilter {
  public static final String ATTR_NAME = "resource_configuration_filters";

  /** The current {@link RuleContext}, used for reporting errors. */
  private final RuleContext ruleContext;

  /**
   * A list of filters that should be applied during the analysis phase. If resources should not be
   * filtered in analysis (e.g., if prefilter_resources = 0), this list should be empty.
   */
  private final ImmutableList<FolderConfiguration> filters;

  /**
   * The raw value of the resource_configuration_filters attribute, as a string of comma-separated
   * qualifier strings. This value is passed directly as input to resource processing utilities that
   * run in the execution phase, so it may represent different qualifiers than those in {@link
   * #filters} if resources should not be pre-filtered during analysis.
   */
  private final String filterString;

  /**
   * @param ruleContext the current {@link RuleContext}, used for reporting errors
   * @param filters a list of filters that should be applied during analysis time. If resources
   *     should not be filtered in analysis (e.g., if prefilter_resources = 0), this list should be
   *     empty.
   * @param filterString the raw value of the resource configuration filters, as a comma-seperated
   *     string
   */
  private ResourceConfigurationFilter(
      RuleContext ruleContext, ImmutableList<FolderConfiguration> filters, String filterString) {
    this.ruleContext = ruleContext;
    this.filters = filters;
    this.filterString = filterString;
  }

  static boolean hasFilters(RuleContext ruleContext) {
    return ruleContext.attributes().isAttributeValueExplicitlySpecified(ATTR_NAME);
  }

  /**
   * Extracts the filters from the current RuleContext, as a string.
   *
   * <p>In BUILD files, filters can be represented as a list of strings, a single comma-seperated
   * string, or a combination of both. This method outputs a single comma-seperated string of
   * filters, which can then be passed directly to resource processing actions.
   *
   * @param ruleContext the current {@link RuleContext}
   * @return the resource configuration filters contained in the {@link RuleContext}, in a single
   *     comma-separated string, or an empty string if no filters exist.
   */
  static String extractFilters(RuleContext ruleContext) {
    if (!hasFilters(ruleContext)) {
      return "";
    }

    return Joiner.on(',').join(ruleContext.getTokenizedStringListAttr(ATTR_NAME));
  }

  static ResourceConfigurationFilter fromRuleContext(RuleContext ruleContext) {
    Preconditions.checkNotNull(ruleContext);

    String resourceConfigurationFilters = extractFilters(ruleContext);

    if (resourceConfigurationFilters.isEmpty()) {
      return empty(ruleContext);
    }

    boolean shouldPrefilter =
        ruleContext.getFragment(AndroidConfiguration.class).useResourcePrefiltering();

    if (!shouldPrefilter) {
      return new ResourceConfigurationFilter(
          ruleContext, ImmutableList.<FolderConfiguration>of(), resourceConfigurationFilters);
    }

    ImmutableList.Builder<FolderConfiguration> builder = ImmutableList.builder();

    for (String filter : resourceConfigurationFilters.split(",")) {
      FolderConfiguration folderConfig = FolderConfiguration.getConfigForQualifierString(filter);

      if (folderConfig == null) {
        ruleContext.attributeError(
            ATTR_NAME, "String '" + filter + "' is not a valid resource configuration filter");
      } else {
        builder.add(folderConfig);
      }
    }

    return new ResourceConfigurationFilter(
        ruleContext, builder.build(), resourceConfigurationFilters);
  }

  static ResourceConfigurationFilter empty(RuleContext ruleContext) {
    return new ResourceConfigurationFilter(
        ruleContext, ImmutableList.<FolderConfiguration>of(), "");
  }

  NestedSet<ResourceContainer> filter(NestedSet<ResourceContainer> resources) {
    if (filters.isEmpty()) {
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
    if (filters.isEmpty()) {
      return artifacts;
    }

    ImmutableList.Builder<Artifact> builder = ImmutableList.builder();

    for (Artifact artifact : artifacts) {
      String containingFolder = artifact.getPath().getParentDirectory().getBaseName();
      if (matches(containingFolder)) {
        builder.add(artifact);
      }
    }

    return builder.build();
  }

  private boolean matches(String containingFolder) {
    FolderConfiguration config = FolderConfiguration.getConfigForFolder(containingFolder);

    if (config == null) {
      ruleContext.ruleError(
          "Resource folder '" + containingFolder + "' has invalid resource qualifiers");

      return true;
    }

    // aapt explicitly ignores the version qualifier; duplicate this behavior here.
    config.setVersionQualifier(VersionQualifier.getQualifier(""));

    for (FolderConfiguration filter : filters) {
      if (config.isMatchFor(filter)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Returns if this object represents an empty filter.
   *
   * <p>Note that non-empty filters are not guaranteed to filter resources during the analysis
   * phase.
   */
  boolean isEmpty() {
    return filterString.isEmpty();
  }

  String getFilterString() {
    return filterString;
  }
}
