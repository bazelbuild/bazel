// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.PrerequisiteMap.Prerequisite;
import com.google.devtools.build.lib.view.config.BuildConfiguration;

import javax.annotation.Nullable;

/**
 * A helper class for building {@link ConfiguredTarget} instances, in particular for non-rule ones.
 * For {@link RuleConfiguredTarget} instances, use {@link RuleContext} instead,
 * which is a subclass of this class.
 *
 * <p>The class is intended to be sub-classed by RuleContext, in order to share the code. However,
 * it's not intended for sub-classing beyond that, and the constructor is intentionally package
 * private to enforce that.
 */
public class TargetContext {

  private final AnalysisEnvironment env;
  private final Target target;
  private final BuildConfiguration configuration;
  private final PrerequisiteMap directPrerequisites;
  private final NestedSet<PackageSpecification> visibility;

  /**
   * The constructor is intentionally package private.
   */
  TargetContext(AnalysisEnvironment env, Target target, BuildConfiguration configuration,
      PrerequisiteMap directPrerequisites,
      NestedSet<PackageSpecification> visibility) {
    this.env = env;
    this.target = target;
    this.configuration = configuration;
    this.directPrerequisites = directPrerequisites;
    this.visibility = visibility;
  }

  public AnalysisEnvironment getAnalysisEnvironment() {
    return env;
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

  /**
   * Returns the host configuration for this rule; keep in mind that there may be multiple different
   * host configurations, even during a single build.
   *
   * @throws IllegalStateException if called for a target that is analyzed in the null configuration
   *         - see {@link #getConfiguration}
   */
  // TODO(bazel-team): Move this to RuleContext, where it's guaranteed to never throw.
  public BuildConfiguration getHostConfiguration() {
    Preconditions.checkState(configuration != null);
    return configuration.getConfiguration(ConfigurationTransition.HOST);
  }

  public NestedSet<PackageSpecification> getVisibility() {
    return visibility;
  }

  TransitiveInfoCollection findDirectPrerequisite(Label label, BuildConfiguration config) {
    Prerequisite prerequisite = directPrerequisites.get(label, config);
    return prerequisite == null ? null : prerequisite.getTransitiveInfoCollection();
  }
}
