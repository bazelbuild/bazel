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
import com.google.devtools.build.lib.actions.PackageRoots;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.skyframe.WalkableGraph;
import javax.annotation.Nullable;

/** Encapsulates the raw analysis result of top level targets and aspects coming from Skyframe. */
public final class SkyframeAnalysisAndExecutionResult extends SkyframeAnalysisResult {
  @Nullable private final DetailedExitCode representativeExecutionExitCode;

  private SkyframeAnalysisAndExecutionResult(
      boolean hasLoadingError,
      boolean hasAnalysisError,
      boolean hasActionConflicts,
      ImmutableSet<ConfiguredTarget> configuredTargets,
      WalkableGraph walkableGraph,
      ImmutableMap<AspectKey, ConfiguredAspect> aspects,
      ImmutableList<TargetAndConfiguration> targetsWithConfiguration,
      PackageRoots packageRoots,
      DetailedExitCode representativeExecutionExitCode) {
    super(
        hasLoadingError,
        hasAnalysisError,
        hasActionConflicts,
        configuredTargets,
        walkableGraph,
        aspects,
        targetsWithConfiguration,
        packageRoots);
    this.representativeExecutionExitCode = representativeExecutionExitCode;
  }

  @Nullable
  public DetailedExitCode getRepresentativeExecutionExitCode() {
    return representativeExecutionExitCode;
  }

  public static SkyframeAnalysisAndExecutionResult success(
      ImmutableSet<ConfiguredTarget> configuredTargets,
      WalkableGraph walkableGraph,
      ImmutableMap<AspectKey, ConfiguredAspect> aspects,
      ImmutableList<TargetAndConfiguration> targetsWithConfiguration,
      PackageRoots packageRoots) {
    return new SkyframeAnalysisAndExecutionResult(
        /* hasLoadingError= */ false,
        /* hasAnalysisError= */ false,
        /* hasActionConflicts= */ false,
        configuredTargets,
        walkableGraph,
        aspects,
        targetsWithConfiguration,
        packageRoots,
        /* representativeExecutionExitCode= */ null);
  }

  public static SkyframeAnalysisAndExecutionResult withErrors(
      boolean hasLoadingError,
      boolean hasAnalysisError,
      boolean hasActionConflicts,
      ImmutableSet<ConfiguredTarget> configuredTargets,
      WalkableGraph walkableGraph,
      ImmutableMap<AspectKey, ConfiguredAspect> aspects,
      ImmutableList<TargetAndConfiguration> targetsWithConfiguration,
      PackageRoots packageRoots,
      @Nullable DetailedExitCode representativeExecutionExitCode) {
    return new SkyframeAnalysisAndExecutionResult(
        hasLoadingError,
        hasAnalysisError,
        hasActionConflicts,
        configuredTargets,
        walkableGraph,
        aspects,
        targetsWithConfiguration,
        packageRoots,
        representativeExecutionExitCode);
  }
}
