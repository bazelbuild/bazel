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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.PackageRoots;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * Return value for {@link com.google.devtools.build.lib.buildtool.AnalysisPhaseRunner}.
 */
public final class AnalysisResult {
  private final BuildConfigurationCollection configurations;
  private final ImmutableSet<ConfiguredTarget> targetsToBuild;
  @Nullable private final ImmutableList<ConfiguredTarget> targetsToTest;
  private final ImmutableSet<ConfiguredTarget> targetsToSkip;
  @Nullable private final String error;
  private final ActionGraph actionGraph;
  private final ArtifactsToOwnerLabels topLevelArtifactsToOwnerLabels;
  private final ImmutableSet<ConfiguredTarget> parallelTests;
  private final ImmutableSet<ConfiguredTarget> exclusiveTests;
  @Nullable private final TopLevelArtifactContext topLevelContext;
  private final ImmutableSet<AspectValue> aspects;
  private final PackageRoots packageRoots;
  private final String workspaceName;
  private final Collection<TargetAndConfiguration> topLevelTargetsWithConfigs;
  private final ImmutableSortedSet<String> nonSymlinkedDirectoriesUnderExecRoot;

  AnalysisResult(
      BuildConfigurationCollection configurations,
      ImmutableSet<ConfiguredTarget> targetsToBuild,
      ImmutableSet<AspectValue> aspects,
      @Nullable ImmutableList<ConfiguredTarget> targetsToTest,
      ImmutableSet<ConfiguredTarget> targetsToSkip,
      @Nullable String error,
      ActionGraph actionGraph,
      ArtifactsToOwnerLabels topLevelArtifactsToOwnerLabels,
      ImmutableSet<ConfiguredTarget> parallelTests,
      ImmutableSet<ConfiguredTarget> exclusiveTests,
      TopLevelArtifactContext topLevelContext,
      PackageRoots packageRoots,
      String workspaceName,
      Collection<TargetAndConfiguration> topLevelTargetsWithConfigs,
      ImmutableSortedSet<String> nonSymlinkedDirectoriesUnderExecRoot) {
    this.configurations = configurations;
    this.targetsToBuild = targetsToBuild;
    this.aspects = aspects;
    this.targetsToTest = targetsToTest;
    this.targetsToSkip = targetsToSkip;
    this.error = error;
    this.actionGraph = actionGraph;
    this.topLevelArtifactsToOwnerLabels = topLevelArtifactsToOwnerLabels;
    this.parallelTests = parallelTests;
    this.exclusiveTests = exclusiveTests;
    this.topLevelContext = topLevelContext;
    this.packageRoots = packageRoots;
    this.workspaceName = workspaceName;
    this.topLevelTargetsWithConfigs = topLevelTargetsWithConfigs;
    this.nonSymlinkedDirectoriesUnderExecRoot = nonSymlinkedDirectoriesUnderExecRoot;
  }

  public BuildConfigurationCollection getConfigurationCollection() {
    return configurations;
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

  /**
   * Returns aspects of configured targets to build.
   *
   * <p>If this list is empty, build the targets returned by {@code getTargetsToBuild()}.
   * Otherwise, only build these aspects of the targets returned by {@code getTargetsToBuild()}.
   */
  public ImmutableSet<AspectValue> getAspects() {
    return aspects;
  }

  /**
   * Returns the configured targets to run as tests, or {@code null} if testing was not requested
   * (e.g. "build" command rather than "test" command).
   */
  @Nullable
  public ImmutableList<ConfiguredTarget> getTargetsToTest() {
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

  public ArtifactsToOwnerLabels getTopLevelArtifactsToOwnerLabels() {
    return topLevelArtifactsToOwnerLabels;
  }

  public ImmutableSet<ConfiguredTarget> getExclusiveTests() {
    return exclusiveTests;
  }

  public ImmutableSet<ConfiguredTarget> getParallelTests() {
    return parallelTests;
  }

  /**
   * Returns an error description (if any).
   */
  @Nullable public String getError() {
    return error;
  }

  public boolean hasError() {
    return error != null;
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

  public String getWorkspaceName() {
    return workspaceName;
  }

  public Collection<TargetAndConfiguration> getTopLevelTargetsWithConfigs() {
    return topLevelTargetsWithConfigs;
  }

  public ImmutableSortedSet<String> getNonSymlinkedDirectoriesUnderExecRoot() {
    return nonSymlinkedDirectoriesUnderExecRoot;
  }

  /**
   * Returns an equivalent {@link AnalysisResult}, except with exclusive tests treated as parallel
   * tests.
   */
  public AnalysisResult withExclusiveTestsAsParallelTests() {
    return new AnalysisResult(
        configurations,
        targetsToBuild,
        aspects,
        targetsToTest,
        targetsToSkip,
        error,
        actionGraph,
        topLevelArtifactsToOwnerLabels,
        Sets.union(parallelTests, exclusiveTests).immutableCopy(),
        /*exclusiveTests=*/ ImmutableSet.of(),
        topLevelContext,
        packageRoots,
        workspaceName,
        topLevelTargetsWithConfigs,
        nonSymlinkedDirectoriesUnderExecRoot);
  }
}
