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
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.PackageRoots;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import java.util.Collection;
import javax.annotation.Nullable;

/** Return value for {@link com.google.devtools.build.lib.buildtool.AnalysisPhaseRunner}. */
public class AnalysisResult {
  private final BuildConfigurationValue configuration;
  private final ImmutableSet<ConfiguredTarget> targetsToBuild;
  @Nullable private final ImmutableSet<ConfiguredTarget> targetsToTest;
  private final ImmutableSet<ConfiguredTarget> targetsToSkip;
  @Nullable private final FailureDetail failureDetail;
  private final ActionGraph actionGraph;
  private final ImmutableSet<Artifact> artifactsToBuild;
  private final ImmutableSet<ConfiguredTarget> parallelTests;
  private final ImmutableSet<ConfiguredTarget> exclusiveTests;
  private final ImmutableSet<ConfiguredTarget> exclusiveIfLocalTests;
  @Nullable private final TopLevelArtifactContext topLevelContext;
  private final ImmutableMap<AspectKey, ConfiguredAspect> aspects;
  private final PackageRoots packageRoots;
  private final String workspaceName;
  private final Collection<TargetAndConfiguration> topLevelTargetsWithConfigs;

  AnalysisResult(
      BuildConfigurationValue configuration,
      ImmutableSet<ConfiguredTarget> targetsToBuild,
      ImmutableMap<AspectKey, ConfiguredAspect> aspects,
      @Nullable ImmutableSet<ConfiguredTarget> targetsToTest,
      ImmutableSet<ConfiguredTarget> targetsToSkip,
      @Nullable FailureDetail failureDetail,
      ActionGraph actionGraph,
      ImmutableSet<Artifact> artifactsToBuild,
      ImmutableSet<ConfiguredTarget> parallelTests,
      ImmutableSet<ConfiguredTarget> exclusiveTests,
      ImmutableSet<ConfiguredTarget> exclusiveIfLocalTests,
      TopLevelArtifactContext topLevelContext,
      PackageRoots packageRoots,
      String workspaceName,
      Collection<TargetAndConfiguration> topLevelTargetsWithConfigs) {
    this.configuration = configuration;
    this.targetsToBuild = targetsToBuild;
    this.aspects = aspects;
    this.targetsToTest = targetsToTest;
    this.targetsToSkip = targetsToSkip;
    this.failureDetail = failureDetail;
    this.actionGraph = actionGraph;
    this.artifactsToBuild = artifactsToBuild;
    this.parallelTests = parallelTests;
    this.exclusiveTests = exclusiveTests;
    this.exclusiveIfLocalTests = exclusiveIfLocalTests;
    this.topLevelContext = topLevelContext;
    this.packageRoots = packageRoots;
    this.workspaceName = workspaceName;
    this.topLevelTargetsWithConfigs = topLevelTargetsWithConfigs;
  }

  public BuildConfigurationValue getConfiguration() {
    return configuration;
  }

  /**
   * Returns configured targets to build.
   */
  public ImmutableSet<ConfiguredTarget> getTargetsToBuild() {
    return targetsToBuild;
  }

  /** @see PackageRoots */
  public PackageRoots getPackageRoots() {
    return packageRoots;
  }

  /** Returns aspects to build. */
  public ImmutableMap<AspectKey, ConfiguredAspect> getAspectsMap() {
    return aspects;
  }

  /**
   * Returns the configured targets to run as tests, or {@code null} if testing was not requested
   * (e.g. "build" command rather than "test" command).
   */
  @Nullable
  public ImmutableSet<ConfiguredTarget> getTargetsToTest() {
    return targetsToTest;
  }

  /**
   * Returns the configured targets that should not be executed because they're not
   * platform-compatible with the current build.
   *
   * <p>For example: tests that aren't intended for the designated CPU.
   */
  public ImmutableSet<ConfiguredTarget> getTargetsToSkip() {
    return targetsToSkip;
  }

  public ImmutableSet<Artifact> getArtifactsToBuild() {
    return artifactsToBuild;
  }

  public ImmutableSet<ConfiguredTarget> getExclusiveTests() {
    return exclusiveTests;
  }

  public ImmutableSet<ConfiguredTarget> getExclusiveIfLocalTests() {
    return exclusiveIfLocalTests;
  }

  public ImmutableSet<ConfiguredTarget> getParallelTests() {
    return parallelTests;
  }

  /** Returns a {@link FailureDetail}, if any failures occurred. */
  @Nullable
  public FailureDetail getFailureDetail() {
    return failureDetail;
  }

  public boolean hasError() {
    return failureDetail != null;
  }

  /**
   * Returns the action graph.
   */
  public ActionGraph getActionGraph() {
    return actionGraph;
  }

  public TopLevelArtifactContext getTopLevelContext() {
    return topLevelContext;
  }

  public Collection<TargetAndConfiguration> getTopLevelTargetsWithConfigs() {
    return topLevelTargetsWithConfigs;
  }

  /**
   * Returns an equivalent {@link AnalysisResult}, except with exclusive tests treated as parallel
   * tests.
   */
  public AnalysisResult withExclusiveTestsAsParallelTests() {
    return new AnalysisResult(
        configuration,
        targetsToBuild,
        aspects,
        targetsToTest,
        targetsToSkip,
        failureDetail,
        actionGraph,
        artifactsToBuild,
        Sets.union(parallelTests, exclusiveTests).immutableCopy(),
        /* exclusiveTests= */ ImmutableSet.of(),
        exclusiveIfLocalTests,
        topLevelContext,
        packageRoots,
        workspaceName,
        topLevelTargetsWithConfigs);
  }

  /**
   * Returns an equivalent {@link AnalysisResult}, except with exclusive tests treated as parallel
   * tests.
   */
  public AnalysisResult withExclusiveIfLocalTestsAsParallelTests() {
    return new AnalysisResult(
        configuration,
        targetsToBuild,
        aspects,
        targetsToTest,
        targetsToSkip,
        failureDetail,
        actionGraph,
        artifactsToBuild,
        Sets.union(parallelTests, exclusiveIfLocalTests).immutableCopy(),
        exclusiveTests,
        /* exclusiveIfLocalTests= */ ImmutableSet.of(),
        topLevelContext,
        packageRoots,
        workspaceName,
        topLevelTargetsWithConfigs);
  }
}
