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

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo.AggregateAttr;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo.Task;
import java.util.EnumMap;
import java.util.Iterator;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Extracts and keeps statistics for one {@link ProfilePhase} for formatting to various outputs.
 */
public final class PhaseStatistics implements Iterable<ProfilerTask> {

  private final ProfilePhase phase;
  private long phaseDurationNanos;
  private long totalDurationNanos;
  private final EnumMap<ProfilerTask, Long> taskDurations;
  private final EnumMap<ProfilerTask, Long> taskCounts;
  private final PhaseVfsStatistics vfsStatistics;
  private boolean wasExecuted;

  public PhaseStatistics(ProfilePhase phase, boolean generateVfsStatistics) {
    this.phase = phase;
    this.taskDurations = new EnumMap<>(ProfilerTask.class);
    this.taskCounts = new EnumMap<>(ProfilerTask.class);
    if (generateVfsStatistics) {
      vfsStatistics = new PhaseVfsStatistics(phase);
    } else {
      vfsStatistics = null;
    }
  }

  public PhaseStatistics(ProfilePhase phase, ProfileInfo info, String workSpaceName, boolean vfs) {
    this(phase, vfs);
    addProfileInfo(workSpaceName, info);
  }

  /**
   * Add statistics from {@link ProfileInfo} to the ones already accumulated for this phase.
   */
  public void addProfileInfo(String workSpaceName, ProfileInfo info) {
    Task phaseTask = info.getPhaseTask(phase);
    if (phaseTask != null) {
      if (vfsStatistics != null) {
        vfsStatistics.addProfileInfo(workSpaceName, info);
      }
      wasExecuted = true;
      long infoPhaseDuration = info.getPhaseDuration(phaseTask);
      phaseDurationNanos += infoPhaseDuration;
      List<Task> taskList = info.getTasksForPhase(phaseTask);
      long duration = infoPhaseDuration;
      for (Task task : taskList) {
        // Tasks on the phaseTask thread already accounted for in the phaseDuration.
        if (task.threadId != phaseTask.threadId) {
          duration += task.durationNanos;
        }
      }
      totalDurationNanos += duration;
      for (ProfilerTask type : ProfilerTask.values()) {
        AggregateAttr attr = info.getStatsForType(type, taskList);
        long totalTime = Math.max(0, attr.totalTime);
        long count = Math.max(0, attr.count);
        add(taskCounts, type, count);
        add(taskDurations, type, totalTime);
      }
    }
  }

  /** Add statistics accumulated in another PhaseStatistics object to this one. */
  public void add(PhaseStatistics other) {
    Preconditions.checkArgument(
        phase == other.phase, "Should not combine statistics from different phases");
    if (other.wasExecuted) {
      if (vfsStatistics != null && other.vfsStatistics != null) {
        vfsStatistics.add(other.vfsStatistics);
      }
      wasExecuted = true;
      phaseDurationNanos += other.phaseDurationNanos;
      totalDurationNanos += other.totalDurationNanos;
      for (ProfilerTask type : other) {
        long otherCount = other.getCount(type);
        long otherDuration = other.getTotalDurationNanos(type);
        add(taskCounts, type, otherCount);
        add(taskDurations, type, otherDuration);
      }
    }
  }

  /** Helper method to sum up long values within an {@link EnumMap}. */
  private static <T extends Enum<T>> void add(EnumMap<T, Long> map, T key, long value) {
    long previous;
    if (map.containsKey(key)) {
      previous = map.get(key);
    } else {
      previous = 0;
    }
    map.put(key, previous + value);
  }

  public ProfilePhase getProfilePhase() {
    return phase;
  }

  @Nullable
  public PhaseVfsStatistics getVfsStatistics() {
    return vfsStatistics;
  }

  /**
   * @return true if no {@link ProfilerTask}s have been executed in this phase, false otherwise
   */
  public boolean isEmpty() {
    return taskCounts.isEmpty();
  }

  /** @return true if the phase was not executed at all, false otherwise */
  public boolean wasExecuted() {
    return wasExecuted;
  }

  /** @return true if a task of the given {@link ProfilerTask} type was executed in this phase */
  public boolean wasExecuted(ProfilerTask taskType) {
    Long count = taskCounts.get(taskType);
    return count != null && count != 0;
  }

  public long getPhaseDurationNanos() {
    return phaseDurationNanos;
  }

  /** @return the sum of all task durations of the given type */
  public long getTotalDurationNanos(ProfilerTask taskType) {
    Long duration = taskDurations.get(taskType);
    if (duration == null) {
      return 0;
    }
    return duration;
  }

  /**
   * @return the average duration of all {@link ProfilerTask}
   */
  public double getMeanDuration(ProfilerTask taskType) {
    if (wasExecuted(taskType)) {
      double duration = taskDurations.get(taskType);
      long count = taskCounts.get(taskType);
      return duration / count;
    }
    return 0;
  }

  /**
   * @return the duration of all {@link ProfilerTask} executed in the phase relative to the total
   *    phase duration
   */
  public double getTotalRelativeDuration(ProfilerTask taskType) {
    Long duration = taskDurations.get(taskType);
    if (duration == null || duration == 0) {
      return 0;
    }
    // sanity check for broken profile files
    Preconditions.checkState(
        totalDurationNanos != 0,
        "Profiler tasks of type %s have non-zero duration %s in phase %s but the phase itself has"
        + " zero duration. Most likely the profile file is broken.",
        taskType,
        duration,
        phase);
    return (double) duration / totalDurationNanos;
  }

  /**
   * @return how many tasks of the given type were executed in this phase
   */
  public long getCount(ProfilerTask taskType) {
    Long count = taskCounts.get(taskType);
    if (count == null) {
      return 0;
    }
    return count;
  }

  /**
   * Iterator over all {@link ProfilerTask}s that were executed at least once and have a total
   * duration greater than 0.
   */
  @Override
  public Iterator<ProfilerTask> iterator() {
    return Iterators.filter(
        taskCounts.keySet().iterator(),
        taskType -> getTotalDurationNanos(taskType) > 0 && wasExecuted(taskType));
  }
}

