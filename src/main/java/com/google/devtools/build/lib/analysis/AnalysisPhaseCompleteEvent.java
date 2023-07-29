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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.TotalAndConfiguredTargetOnlyMetric;
import com.google.devtools.build.lib.pkgcache.PackageManager.PackageManagerStatistics;
import java.util.Collection;

/**
 * This event is fired after the analysis phase is complete.
 */
public class AnalysisPhaseCompleteEvent {

  private final Collection<ConfiguredTarget> topLevelTargets;
  private final long timeInMs;
  private final TotalAndConfiguredTargetOnlyMetric targetsConfigured;
  private final PackageManagerStatistics pkgManagerStats;
  private final TotalAndConfiguredTargetOnlyMetric actionsConstructed;
  private final boolean analysisCacheDropped;

  public AnalysisPhaseCompleteEvent(
      Collection<? extends ConfiguredTarget> topLevelTargets,
      TotalAndConfiguredTargetOnlyMetric targetsConfigured,
      TotalAndConfiguredTargetOnlyMetric actionsConstructed,
      long timeInMs,
      PackageManagerStatistics pkgManagerStats,
      boolean analysisCacheDropped) {
    this(
        topLevelTargets,
        targetsConfigured,
        actionsConstructed,
        timeInMs,
        pkgManagerStats,
        analysisCacheDropped,
        /*skymeldEnabled=*/ false);
  }

  private AnalysisPhaseCompleteEvent(
      Collection<? extends ConfiguredTarget> topLevelTargets,
      TotalAndConfiguredTargetOnlyMetric targetsConfigured,
      TotalAndConfiguredTargetOnlyMetric actionsConstructed,
      long timeInMs,
      PackageManagerStatistics pkgManagerStats,
      boolean analysisCacheDropped,
      boolean skymeldEnabled) {
    this.timeInMs = timeInMs;
    this.topLevelTargets = ImmutableList.copyOf(topLevelTargets);
    this.targetsConfigured = checkNotNull(targetsConfigured);
    this.pkgManagerStats = pkgManagerStats;
    this.actionsConstructed = checkNotNull(actionsConstructed);
    this.analysisCacheDropped = analysisCacheDropped;
  }

  /**
   * A factory method for the AnalysisPhaseCompleteEvent that originates from Skymeld.
   *
   * <p>This marks the end of the analysis-related work within the build. Contrary to the
   * traditional build where there is a distinct separation between the loading/analysis and
   * execution phases, overlapping is possible with Skymeld. We are likely already deep into action
   * execution when this event is posted.
   */
  public static AnalysisPhaseCompleteEvent fromSkymeld(
      Collection<? extends ConfiguredTarget> topLevelTargets,
      TotalAndConfiguredTargetOnlyMetric targetsConfigured,
      TotalAndConfiguredTargetOnlyMetric actionsConstructed,
      long timeInMs,
      PackageManagerStatistics pkgManagerStats,
      boolean analysisCacheDropped) {
    return new AnalysisPhaseCompleteEvent(
        topLevelTargets,
        targetsConfigured,
        actionsConstructed,
        timeInMs,
        pkgManagerStats,
        analysisCacheDropped,
        /*skymeldEnabled=*/ true);
  }

  /**
   * Returns the set of active topLevelTargets remaining, which is a subset of the topLevelTargets
   * we attempted to analyze.
   */
  public Collection<ConfiguredTarget> getTopLevelTargets() {
    return topLevelTargets;
  }

  /** Returns the number of targets/aspects configured during analysis. */
  public TotalAndConfiguredTargetOnlyMetric getTargetsConfigured() {
    return targetsConfigured;
  }

  public long getTimeInMs() {
    return timeInMs;
  }

  /** Returns the actions constructed during this analysis. */
  public TotalAndConfiguredTargetOnlyMetric getActionsConstructed() {
    return actionsConstructed;
  }

  public boolean wasAnalysisCacheDropped() {
    return analysisCacheDropped;
  }

  /**
   * Returns package manager statistics.
   */
  public PackageManagerStatistics getPkgManagerStats() {
    return pkgManagerStats;
  }
}
