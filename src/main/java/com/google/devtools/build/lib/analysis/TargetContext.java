// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.base.Objects;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimaps;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A helper class for building {@link ConfiguredTarget} instances, in particular for non-rule ones.
 * For {@link com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget}
 * instances, use {@link RuleContext} instead, which is a subclass of this class.
 *
 * <p>The class is intended to be sub-classed by RuleContext, in order to share the code. However,
 * it's not intended for sub-classing beyond that, and the constructor is intentionally package
 * private to enforce that.
 */
public class TargetContext {

  private final AnalysisEnvironment env;
  private final Target target;
  private final BuildConfiguration configuration;

  /**
   * This only contains prerequisites that are not declared in rule attributes, with the exception
   * of visibility (i.e., visibility is represented here, even though it is a rule attribute in case
   * of a rule). Rule attributes are handled by the {@link RuleContext} subclass.
   */
  private final ListMultimap<Label, ConfiguredTargetAndData> directPrerequisites;

  private final NestedSet<PackageGroupContents> visibility;

  /**
   * The constructor is intentionally package private.
   *
   * <p>directPrerequisites is expected to be ordered.
   */
  TargetContext(
      AnalysisEnvironment env,
      Target target,
      BuildConfiguration configuration,
      Set<ConfiguredTargetAndData> directPrerequisites,
      NestedSet<PackageGroupContents> visibility) {
    this.env = env;
    this.target = target;
    this.configuration = configuration;
    this.directPrerequisites =
        Multimaps.index(directPrerequisites, prereq -> prereq.getTarget().getLabel());
    this.visibility = visibility;
  }

  public AnalysisEnvironment getAnalysisEnvironment() {
    return env;
  }

  public ActionKeyContext getActionKeyContext() {
    return env.getActionKeyContext();
  }

  public Target getTarget() {
    return target;
  }

  public Label getLabel() {
    return target.getLabel();
  }

  /**
   * Returns the configuration for this target. This may return null if the target is supposed to be
   * configuration-independent (like an input file, or a visibility rule). However, this is
   * guaranteed to be non-null for rules and for output files.
   */
  @Nullable
  public BuildConfiguration getConfiguration() {
    return configuration;
  }

  public BuildConfigurationValue.Key getConfigurationKey() {
    return BuildConfigurationValue.key(configuration);
  }

  public NestedSet<PackageGroupContents> getVisibility() {
    return visibility;
  }

  /**
   * Returns the prerequisite with the given label and configuration, or null if no such
   * prerequisite exists. If configuration is absent, return the first prerequisite with the given
   * label.
   */
  @Nullable
  public TransitiveInfoCollection findDirectPrerequisite(
      Label label, Optional<BuildConfiguration> config) {
    if (directPrerequisites.containsKey(label)) {
      List<ConfiguredTargetAndData> prerequisites = directPrerequisites.get(label);
      // If the config is present, find the prereq with that configuration. Otherwise, return the
      // first.
      if (!config.isPresent()) {
        if (prerequisites.isEmpty()) {
          return null;
        }
        return Iterables.getFirst(prerequisites, null).getConfiguredTarget();
      }
      for (ConfiguredTargetAndData prerequisite : prerequisites) {
        if (Objects.equal(prerequisite.getConfiguration(), config.get())) {
          return prerequisite.getConfiguredTarget();
        }
      }
    }
    return null;
  }
}
