// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler.statistics;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.TraceEvent;
import java.time.Duration;
import java.util.List;

/**
 * Keeps a predefined list of {@link TraceEvent}'s cumulative durations and allows iterating over
 * pairs of their descriptions and relative durations.
 */
public final class CriticalPathStatistics {
  private final ImmutableList<TraceEvent> criticalPathEntries;
  private Duration totalDuration = Duration.ZERO;

  public CriticalPathStatistics(List<TraceEvent> traceEvents) {
    ImmutableList.Builder<TraceEvent> criticalPathEntriesBuilder = new ImmutableList.Builder<>();
    for (TraceEvent traceEvent : traceEvents) {
      if (ProfilerTask.CRITICAL_PATH_COMPONENT.description.equals(traceEvent.category())) {
        criticalPathEntriesBuilder.add(traceEvent);
        totalDuration = totalDuration.plus(traceEvent.duration());
      }
    }
    this.criticalPathEntries = criticalPathEntriesBuilder.build();
  }

  public Duration getTotalDuration() {
    return totalDuration;
  }

  public ImmutableList<TraceEvent> getCriticalPathEntries() {
    return criticalPathEntries;
  }
}

