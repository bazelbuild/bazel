// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.util.DetailedExitCode;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * Return value for {@link com.google.devtools.build.lib.buildtool.AnalysisAndExecutionPhaseRunner}.
 * This is meant to be the drop-in replacement for AnalysisResult later on. This is part of
 * https://github.com/bazelbuild/bazel/issues/14057. Internal: b/147350683.
 */
public final class AnalysisAndExecutionResult extends AnalysisResult {
  private final DetailedExitCode executionDetailedExitCode;

  AnalysisAndExecutionResult(
      BuildConfigurationValue configuration,
      ImmutableSet<ConfiguredTarget> targetsToBuild,
      ImmutableMap<AspectKey, ConfiguredAspect> aspects,
      @Nullable ImmutableSet<ConfiguredTarget> targetsToTest,
      ImmutableSet<ConfiguredTarget> targetsToSkip,
      @Nullable FailureDetail analysisFailureDetail,
      @Nullable DetailedExitCode executionDetailedExitCode,
      ImmutableSet<Artifact> artifactsToBuild,
      ImmutableSet<ConfiguredTarget> parallelTests,
      ImmutableSet<ConfiguredTarget> exclusiveTests,
      ImmutableSet<ConfiguredTarget> exclusiveIfLocalTests,
      TopLevelArtifactContext topLevelContext,
      String workspaceName,
      Collection<TargetAndConfiguration> topLevelTargetsWithConfigs) {
    super(
        configuration,
        targetsToBuild,
        aspects,
        targetsToTest,
        targetsToSkip,
        analysisFailureDetail,
        /* actionGraph= */ null,
        artifactsToBuild,
        parallelTests,
        exclusiveTests,
        exclusiveIfLocalTests,
        topLevelContext,
        /* packageRoots= */ null,
        workspaceName,
        topLevelTargetsWithConfigs);
    this.executionDetailedExitCode = executionDetailedExitCode;
  }

  @Nullable
  public DetailedExitCode getExecutionDetailedExitCode() {
    return executionDetailedExitCode;
  }
}
