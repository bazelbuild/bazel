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

import static com.google.devtools.build.lib.pkgcache.PackageManager.PackageManagerStatistics;

import com.google.common.collect.ImmutableList;
import java.util.Collection;

/**
 * This event is fired after the analysis phase is complete.
 */
public class AnalysisPhaseCompleteEvent {

  private final Collection<ConfiguredTarget> topLevelTargets;
  private final long timeInMs;
  private int targetsLoaded;
  private int targetsConfigured;
  private final PackageManagerStatistics pkgManagerStats;
  private final int actionsConstructed;
  private final boolean analysisCacheDropped;

  /**
   * Construct the event.
   *
   * @param topLevelTargets The set of active topLevelTargets that remain.
   */
  public AnalysisPhaseCompleteEvent(
      Collection<? extends ConfiguredTarget> topLevelTargets,
      int targetsLoaded,
      int targetsConfigured,
      long timeInMs,
      PackageManagerStatistics pkgManagerStats,
      int actionsConstructed,
      boolean analysisCacheDropped) {
    this.timeInMs = timeInMs;
    this.topLevelTargets = ImmutableList.copyOf(topLevelTargets);
    this.targetsLoaded = targetsLoaded;
    this.targetsConfigured = targetsConfigured;
    this.pkgManagerStats = pkgManagerStats;
    this.actionsConstructed = actionsConstructed;
    this.analysisCacheDropped = analysisCacheDropped;
  }

  /**
   * Returns the set of active topLevelTargets remaining, which is a subset of the topLevelTargets
   * we attempted to analyze.
   */
  public Collection<ConfiguredTarget> getTopLevelTargets() {
    return topLevelTargets;
  }

  /** Returns the number of targets loaded during analysis */
  public int getTargetsLoaded() {
    return targetsLoaded;
  }

  /** Returns the number of targets configured during analysis */
  public int getTargetsConfigured() {
    return targetsConfigured;
  }

  public long getTimeInMs() {
    return timeInMs;
  }

  public int getActionsConstructed() {
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
