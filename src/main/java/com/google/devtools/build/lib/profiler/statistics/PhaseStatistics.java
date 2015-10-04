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

import com.google.common.base.Predicate;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.profiler.ProfileInfo;
import com.google.devtools.build.lib.profiler.ProfileInfo.AggregateAttr;
import com.google.devtools.build.lib.profiler.ProfileInfo.Task;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.ProfilerTask;

import java.util.EnumMap;
import java.util.Iterator;
import java.util.List;

/**
 * Extracts and keeps statistics for one {@link ProfilePhase} for formatting to various outputs.
 */
public final class PhaseStatistics implements Iterable<ProfilerTask> {

  private final ProfilePhase phase;
  private final long phaseDurationNanos;
  private final long totalDurationNanos;
  private final EnumMap<ProfilerTask, AggregateAttr> aggregateTaskStatistics;
  private final PhaseVfsStatistics vfsStatistics;
  private final boolean wasExecuted;

  public PhaseStatistics(ProfilePhase phase, ProfileInfo info, String workSpaceName) {
    this.phase = phase;
    this.aggregateTaskStatistics = new EnumMap<>(ProfilerTask.class);
    Task phaseTask = info.getPhaseTask(phase);
    vfsStatistics = new PhaseVfsStatistics(workSpaceName, phase, info);
    if (phaseTask == null) {
      wasExecuted = false;
      totalDurationNanos = 0;
      phaseDurationNanos = 0;
    } else {
      wasExecuted = true;
      phaseDurationNanos = info.getPhaseDuration(phaseTask);
      List<Task> taskList = info.getTasksForPhase(phaseTask);
      long duration = phaseDurationNanos;
      for (Task task : taskList) {
        // Tasks on the phaseTask thread already accounted for in the phaseDuration.
        if (task.threadId != phaseTask.threadId) {
          duration += task.durationNanos;
        }
      }
      totalDurationNanos = duration;
      for (ProfilerTask type : ProfilerTask.values()) {
        aggregateTaskStatistics.put(type, info.getStatsForType(type, taskList));
      }
    }
  }

  public ProfilePhase getProfilePhase() {
    return phase;
  }

  public PhaseVfsStatistics getVfsStatistics() {
    return vfsStatistics;
  }

  /**
   * @return true if no {@link ProfilerTask}s have been executed in this phase, false otherwise
   */
  public boolean isEmpty() {
    return aggregateTaskStatistics.isEmpty();
  }

  /**
   * @return true if the phase was not executed at all, false otherwise
   */
  public boolean wasExecuted() {
    return wasExecuted;
  }

  public long getPhaseDurationNanos() {
    return phaseDurationNanos;
  }

  public long getTotalDurationNanos() {
    return totalDurationNanos;
  }

  /**
   * @return true if a task of the given {@link ProfilerTask} type was executed in this phase
   */
  public boolean wasExecuted(ProfilerTask taskType) {
    return aggregateTaskStatistics.get(taskType).count != 0;
  }

  /**
   * @return the sum of all task durations of the given type
   */
  public long getTotalDurationNanos(ProfilerTask taskType) {
    return aggregateTaskStatistics.get(taskType).totalTime;
  }

  /**
   * @return the average duration of all {@link ProfilerTask}
   */
  public double getMeanDuration(ProfilerTask taskType) {
    if (wasExecuted(taskType)) {
      AggregateAttr stats = aggregateTaskStatistics.get(taskType);
      return (double) stats.totalTime / stats.count;
    }
    return 0;
  }

  /**
   * @return the duration of all {@link ProfilerTask} executed in the phase relative to the total
   *    phase duration
   */
  public double getTotalRelativeDuration(ProfilerTask taskType) {
    if (wasExecuted(taskType)) {
      return (double) aggregateTaskStatistics.get(taskType).totalTime / totalDurationNanos;
    }
    return 0;
  }

  /**
   * @return how many tasks of the given type were executed in this phase
   */
  public int getCount(ProfilerTask taskType) {
    return aggregateTaskStatistics.get(taskType).count;
  }

  /**
   * Iterator over all {@link ProfilerTask}s that were executed at least once and have a total
   * duration greater than 0.
   */
  @Override
  public Iterator<ProfilerTask> iterator() {
    return Iterators.filter(
        aggregateTaskStatistics.keySet().iterator(),
        new Predicate<ProfilerTask>() {
          @Override
          public boolean apply(ProfilerTask taskType) {

            return getTotalDurationNanos(taskType) != 0 && getCount(taskType) != 0;
          }
        });
  }
}

