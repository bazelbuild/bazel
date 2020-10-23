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
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import java.util.List;

/**
 * A helper class for compilation helpers.
 */
public final class CompilationHelper {
  /**
   * Returns a middleman for all files to build for the given configured target.
   * If multiple calls are made, then it returns the same artifact for
   * configurations with the same internal directory.
   *
   * <p>The resulting middleman only aggregates the inputs and must be expanded
   * before an action using it can be sent to a distributed execution-system.
   */
  public static NestedSet<Artifact> getAggregatingMiddleman(
      RuleContext ruleContext, String purpose, NestedSet<Artifact> filesToBuild) {
    return NestedSetBuilder.wrap(Order.STABLE_ORDER, getMiddlemanInternal(
        ruleContext.getAnalysisEnvironment(), ruleContext, ruleContext.getActionOwner(), purpose,
        filesToBuild));
  }

  /**
   * Internal implementation for getAggregatingMiddleman / getAggregatingMiddlemanWithSolibSymlinks.
   */
  private static List<Artifact> getMiddlemanInternal(AnalysisEnvironment env,
      RuleContext ruleContext, ActionOwner actionOwner, String purpose,
      NestedSet<Artifact> filesToBuild) {
    if (filesToBuild == null) {
      return ImmutableList.of();
    }
    MiddlemanFactory factory = env.getMiddlemanFactory();
    return ImmutableList.of(
        factory.createMiddlemanAllowMultiple(
            env,
            actionOwner,
            ruleContext.getPackageDirectory(),
            purpose,
            filesToBuild,
            ruleContext.getMiddlemanDirectory()));
  }

  // TODO(bazel-team): remove this duplicated code after the ConfiguredTarget migration
  /**
   * Returns a middleman for all files to build for the given configured target.
   * If multiple calls are made, then it returns the same artifact for
   * configurations with the same internal directory.
   *
   * <p>The resulting middleman only aggregates the inputs and must be expanded
   * before an action using it can be sent to a distributed execution-system.
   */
  public static NestedSet<Artifact> getAggregatingMiddleman(
      RuleContext ruleContext, String purpose, TransitiveInfoCollection dep) {
    return NestedSetBuilder.wrap(Order.STABLE_ORDER, getMiddlemanInternal(
        ruleContext.getAnalysisEnvironment(), ruleContext, ruleContext.getActionOwner(), purpose,
        dep));
  }

  /**
   * Internal implementation for getAggregatingMiddleman / getAggregatingMiddlemanWithSolibSymlinks.
   */
  private static List<Artifact> getMiddlemanInternal(AnalysisEnvironment env,
      RuleContext ruleContext, ActionOwner actionOwner, String purpose,
      TransitiveInfoCollection dep) {
    if (dep == null) {
      return ImmutableList.of();
    }
    MiddlemanFactory factory = env.getMiddlemanFactory();
    NestedSet<Artifact> artifacts = dep.getProvider(FileProvider.class).getFilesToBuild();
    return ImmutableList.of(
        factory.createMiddlemanAllowMultiple(
            env,
            actionOwner,
            ruleContext.getPackageDirectory(),
            purpose,
            artifacts,
            ruleContext.getMiddlemanDirectory()));
  }
}
