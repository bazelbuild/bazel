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

import com.google.common.collect.Maps;
import com.google.common.collect.Multimaps;
import com.google.common.collect.Ordering;
import com.google.common.collect.TreeMultimap;
import com.google.devtools.build.lib.profiler.ProfileInfo;
import com.google.devtools.build.lib.profiler.ProfileInfo.Task;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.ProfilerTask;

import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Compute and store statistics of all {@link ProfilerTask}s that begin with VFS_ in sorted order.
 */
public final class PhaseVfsStatistics implements Iterable<ProfilerTask> {

  /**
   * Pair of duration and count for sorting by duration first and count in case of tie
   */
  public static class Stat implements Comparable<Stat> {
    public long duration;
    public long count;

    @Override
    public int compareTo(Stat o) {
      return this.duration == o.duration
          ? Long.compare(this.count, o.count)
          : Long.compare(this.duration, o.duration);
    }
  }

  private final ProfilePhase phase;
  private final EnumMap<ProfilerTask, TreeMultimap<Stat, String>> sortedStatistics;
  private final String workSpaceName;

  public PhaseVfsStatistics(final String workSpaceName, ProfilePhase phase, ProfileInfo info) {
    this.workSpaceName = workSpaceName;
    this.phase = phase;
    this.sortedStatistics = Maps.newEnumMap(ProfilerTask.class);

    Task phaseTask = info.getPhaseTask(phase);
    if (phaseTask == null) {
      return;
    }
    collectVfsEntries(info.getTasksForPhase(phaseTask));
  }

  public ProfilePhase getProfilePhase() {
    return phase;
  }

  public boolean isEmpty() {
    return sortedStatistics.isEmpty();
  }

  public Iterable<Entry<Stat, String>> getSortedStatistics(ProfilerTask taskType) {
    return sortedStatistics.get(taskType).entries();
  }

  public int getStatisticsCount(ProfilerTask taskType) {
    return sortedStatistics.get(taskType).size();
  }

  @Override
  public Iterator<ProfilerTask> iterator() {
    return sortedStatistics.keySet().iterator();
  }

  /**
   * Group into VFS operations and build maps from path to duration.
   */
  private void collectVfsEntries(List<Task> taskList) {
    EnumMap<ProfilerTask, Map<String, Stat>> stats = Maps.newEnumMap(ProfilerTask.class);
    for (Task task : taskList) {
      collectVfsEntries(Arrays.asList(task.subtasks));
      if (!task.type.name().startsWith("VFS_")) {
        continue;
      }

      Map<String, Stat> statsForType = stats.get(task.type);
      if (statsForType == null) {
        statsForType = new HashMap<>();
        stats.put(task.type, statsForType);
      }

      String path = currentPathMapping(task.getDescription());

      Stat stat = statsForType.get(path);
      if (stat == null) {
        stat = new Stat();
      }

      stat.duration += task.durationNanos;
      stat.count++;
      statsForType.put(path, stat);
    }
    // Reverse the maps to get maps from duration to path. We use a TreeMultimap to sort by
    // duration and because durations are not unique.
    for (ProfilerTask type : stats.keySet()) {
      Map<String, Stat> statsForType = stats.get(type);
      TreeMultimap<Stat, String> sortedStats =
          TreeMultimap.create(Ordering.natural().reverse(), Ordering.natural());

      Multimaps.invertFrom(Multimaps.forMap(statsForType), sortedStats);
      sortedStatistics.put(type, sortedStats);
    }
  }

  private String currentPathMapping(String input) {
    if (workSpaceName.isEmpty()) {
      return input;
    } else {
      return input.substring(input.lastIndexOf("/" + workSpaceName) + 1);
    }
  }
}

