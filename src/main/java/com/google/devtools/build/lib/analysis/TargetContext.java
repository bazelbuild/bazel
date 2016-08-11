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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.Target;

import java.util.List;

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
  /**
   * This list only contains prerequisites that are not declared in rule attributes, with the
   * exception of visibility (i.e., visibility is represented here, even though it is a rule
   * attribute in case of a rule). Rule attributes are handled by the {@link RuleContext} subclass.
   */
  private final List<ConfiguredTarget> directPrerequisites;
  private final NestedSet<PackageSpecification> visibility;

  /**
   * The constructor is intentionally package private.
   *
   * <p>directPrerequisites is expected to be ordered.
   */
  TargetContext(AnalysisEnvironment env, Target target, BuildConfiguration configuration,
      Iterable<ConfiguredTarget> directPrerequisites,
      NestedSet<PackageSpecification> visibility) {
    this.env = env;
    this.target = target;
    this.configuration = configuration;
    this.directPrerequisites = ImmutableList.<ConfiguredTarget>copyOf(directPrerequisites);
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

  public NestedSet<PackageSpecification> getVisibility() {
    return visibility;
  }

  TransitiveInfoCollection findDirectPrerequisite(Label label, BuildConfiguration config) {
    for (ConfiguredTarget prerequisite : directPrerequisites) {
      if (prerequisite.getLabel().equals(label) && (prerequisite.getConfiguration() == config)) {
        return prerequisite;
      }
    }
    return null;
  }
}
