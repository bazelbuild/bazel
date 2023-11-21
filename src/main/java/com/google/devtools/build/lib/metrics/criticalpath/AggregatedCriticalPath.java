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

package com.google.devtools.build.lib.metrics.criticalpath;

import static com.google.devtools.build.lib.clock.BlazeClock.formatTime;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.AggregatedSpawnMetrics;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.Locale;

/**
 * Aggregates all the critical path components in one object. This allows us to easily access the
 * components data and have a proper toString().
 */
@SuppressWarnings("GoodTime") // Use ints instead of Durations to improve build time (cl/505728570)
public class AggregatedCriticalPath {
  public static final AggregatedCriticalPath EMPTY =
      new AggregatedCriticalPath(0, AggregatedSpawnMetrics.EMPTY, ImmutableList.of());

  private final Duration totalTime;
  private final AggregatedSpawnMetrics aggregatedSpawnMetrics;
  private final ImmutableList<CriticalPathComponent> criticalPathComponents;

  public AggregatedCriticalPath(
      int totalTimeInMs,
      AggregatedSpawnMetrics aggregatedSpawnMetrics,
      ImmutableList<CriticalPathComponent> criticalPathComponents) {
    this.totalTime = Duration.ofMillis(totalTimeInMs);
    this.aggregatedSpawnMetrics = aggregatedSpawnMetrics;
    this.criticalPathComponents = criticalPathComponents;
  }

  /** Total wall time spent running the critical path actions. */
  @VisibleForTesting
  public Duration totalTime() {
    return totalTime;
  }

  public AggregatedSpawnMetrics getSpawnMetrics() {
    return aggregatedSpawnMetrics;
  }

  /** Returns a list of all the component stats for the critical path. */
  public ImmutableList<CriticalPathComponent> components() {
    return criticalPathComponents;
  }

  public String getNewStringSummary() {
    Duration executionWallTime =
        aggregatedSpawnMetrics.getTotalDuration(
            SpawnMetrics::executionWallTimeInMs, ChronoUnit.MILLIS);
    Duration overheadTime =
        aggregatedSpawnMetrics
            .getTotalDuration(SpawnMetrics::totalTimeInMs, ChronoUnit.MILLIS)
            .minus(executionWallTime);
    return String.format(
        Locale.US,
        "Execution critical path %s (setup %s, action wall time %s)",
        formatTime(totalTime),
        formatTime(overheadTime),
        formatTime(executionWallTime));
  }

  @Override
  public String toString() {
    return toString(false, true);
  }

  private String toString(boolean summary, boolean remote) {
    StringBuilder sb = new StringBuilder("Critical Path: ");
    sb.append(String.format("%s", formatTime(totalTime)));
    if (remote) {
      sb.append(", ");
      sb.append(getSpawnMetrics().toString(totalTime, summary));
    }
    if (summary || criticalPathComponents.isEmpty()) {
      return sb.toString();
    }
    sb.append("\n  ");
    Joiner.on("\n  ").appendTo(sb, criticalPathComponents);
    return sb.toString();
  }

  /**
   * Returns a summary version of the critical path stats that omits stats that are not useful to
   * the user.
   */
  public String toStringSummary() {
    return toString(true, true);
  }

  /**
   * Same as toStringSummary but also omits remote stats. This is to be used in Bazel because
   * currently the Remote stats are not calculated correctly.
   */
  public String toStringSummaryNoRemote() {
    return toString(true, false);
  }
}
