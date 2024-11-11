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
import com.google.common.collect.ImmutableMap;
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
  private final ImmutableMap<String, Integer> actionsConstructedByMnemonic;
  private final boolean analysisCacheDropped;

  public AnalysisPhaseCompleteEvent(
      Collection<? extends ConfiguredTarget> topLevelTargets,
      TotalAndConfiguredTargetOnlyMetric targetsConfigured,
      TotalAndConfiguredTargetOnlyMetric actionsConstructed,
      ImmutableMap<String, Integer> actionsConstructedByMnemonic,
      long timeInMs,
      PackageManagerStatistics pkgManagerStats,
      boolean analysisCacheDropped) {
    this.timeInMs = timeInMs;
    this.topLevelTargets = ImmutableList.copyOf(topLevelTargets);
    this.targetsConfigured = checkNotNull(targetsConfigured);
    this.pkgManagerStats = pkgManagerStats;
    this.actionsConstructed = checkNotNull(actionsConstructed);
    this.actionsConstructedByMnemonic = checkNotNull(actionsConstructedByMnemonic);
    this.analysisCacheDropped = analysisCacheDropped;
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

  public ImmutableMap<String, Integer> getActionsConstructedByMnemonic() {
    return actionsConstructedByMnemonic;
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
