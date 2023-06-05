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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.PackageRoots;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.skyframe.WalkableGraph;

/**
 *  Encapsulates the raw analysis result of top level targets and aspects coming from Skyframe.
 */
public class SkyframeAnalysisResult {
  private final boolean hasLoadingError;
  private final boolean hasAnalysisError;
  private final boolean hasActionConflicts;
  private final ImmutableSet<ConfiguredTarget> configuredTargets;
  private final WalkableGraph walkableGraph;
  private final ImmutableMap<AspectKey, ConfiguredAspect> aspects;
  private final ImmutableList<TargetAndConfiguration> targetsWithConfiguration;
  private final PackageRoots packageRoots;

  SkyframeAnalysisResult(
      boolean hasLoadingError,
      boolean hasAnalysisError,
      boolean hasActionConflicts,
      ImmutableSet<ConfiguredTarget> configuredTargets,
      WalkableGraph walkableGraph,
      ImmutableMap<AspectKey, ConfiguredAspect> aspects,
      ImmutableList<TargetAndConfiguration> targetsWithConfiguration,
      PackageRoots packageRoots) {
    this.hasLoadingError = hasLoadingError;
    this.hasAnalysisError = hasAnalysisError;
    this.hasActionConflicts = hasActionConflicts;
    this.configuredTargets = configuredTargets;
    this.walkableGraph = walkableGraph;
    this.aspects = aspects;
    this.targetsWithConfiguration = targetsWithConfiguration;
    this.packageRoots = packageRoots;
  }

  /**
   * If the new simplified loading phase is enabled, then we can also see loading errors during the
   * analysis phase. This method returns true if any such errors were encountered. However, you also
   * always need to check if the loading result has an error! These will be merged eventually.
   */
  public boolean hasLoadingError() {
    return hasLoadingError;
  }

  public boolean hasAnalysisError() {
    return hasAnalysisError;
  }

  public boolean hasActionConflicts() {
    return hasActionConflicts;
  }

  public ImmutableSet<ConfiguredTarget> getConfiguredTargets() {
    return configuredTargets;
  }

  public WalkableGraph getWalkableGraph() {
    return walkableGraph;
  }

  public ImmutableMap<AspectKey, ConfiguredAspect> getAspects() {
    return aspects;
  }

  public ImmutableList<TargetAndConfiguration> getTargetsWithConfiguration() {
    return targetsWithConfiguration;
  }

  public PackageRoots getPackageRoots() {
    return packageRoots;
  }

  /**
   * Returns an equivalent {@link SkyframeAnalysisResult}, except with errored targets removed from
   * the configured target list.
   */
  public SkyframeAnalysisResult withAdditionalErroredTargets(
      ImmutableSet<ConfiguredTarget> erroredTargets) {
    return new SkyframeAnalysisResult(
        hasLoadingError,
        /* hasAnalysisError= */ true,
        hasActionConflicts,
        Sets.difference(configuredTargets, erroredTargets).immutableCopy(),
        walkableGraph,
        aspects,
        targetsWithConfiguration,
        packageRoots);
  }
}
