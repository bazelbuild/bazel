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

import static com.google.devtools.build.lib.profiler.ProfilerTask.CRITICAL_PATH;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo.Task;
import java.time.Duration;
import java.util.ArrayList;

/**
 * Keeps a predefined list of {@link Task}'s cumulative durations and allows iterating over pairs of
 * their descriptions and relative durations.
 */
public final class CriticalPathStatistics {
  private final ImmutableList<Task> criticalPathEntries;
  private Duration totalDuration = Duration.ZERO;

  public CriticalPathStatistics(ProfileInfo info) {
    ArrayList<Task> criticalPathEntries = new ArrayList<>();
    for (Task task : info.rootTasksById) {
      if (task.type == CRITICAL_PATH) {
        for (Task criticalPathEntry : task.subtasks) {
          totalDuration = totalDuration.plus(Duration.ofNanos(criticalPathEntry.durationNanos));
          criticalPathEntries.add(criticalPathEntry);
        }
      }
    }
    this.criticalPathEntries = ImmutableList.copyOf(criticalPathEntries);
  }

  public Duration getTotalDuration() {
    return totalDuration;
  }

  public ImmutableList<Task> getCriticalPathEntries() {
    return criticalPathEntries;
  }
}

